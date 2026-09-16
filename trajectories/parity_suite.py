import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Environment must be configured before importing minisweagent modules.
os.environ.setdefault("MSWEA_CONFIGURED", "true")
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("OPENAI_API_KEY", "sk-6b1e1cd0349b6e364252ec79db59318aaa98835a2dabb955")
os.environ.setdefault("OPENAI_BASE_URL", "http://172.18.12.5:18080/v1")
os.environ.setdefault("OPENAI_API_BASE", "http://172.18.12.5:18080/v1")
os.environ.setdefault("MSWEA_MODEL_NAME", "openai/deepseek-v4.1-flash")

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.config import builtin_config_dir  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUST_EXE = ROOT / "rust" / "target" / "debug" / "mini.exe"
SUITE_DIR = ROOT / "trajectories" / "suite"
WORKSPACES = SUITE_DIR / "workspaces"

CASES = [
    {
        "name": "file_roundtrip",
        "task": """You are in a Windows command shell. Perform exactly these steps:
1. Run `echo alpha > case_file.txt`.
2. Run `echo beta >> case_file.txt`.
3. Run `type case_file.txt`.
4. Finish by running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
Do not run any other commands.""",
    },
    {
        "name": "python_compute",
        "task": """You are in a Windows command shell. Perform exactly these steps:
1. Run `python -c "print(6 * 7)"`.
2. Finish by running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
Do not run any other commands.""",
    },
    {
        "name": "error_recovery",
        "task": """You are in a Windows command shell. Perform exactly these steps:
1. Run `dir this_file_does_not_exist_12345`.
2. Run `echo recovered`.
3. Finish by running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
Do not run any other commands.""",
    },
    {
        "name": "multi_file",
        "task": """You are in a Windows command shell. Perform exactly these steps:
1. Run `mkdir case_dir`.
2. Run `echo one > case_dir\\one.txt`.
3. Run `echo two > case_dir\\two.txt`.
4. Run `type case_dir\\one.txt && type case_dir\\two.txt`.
5. Finish by running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

IMPORTANT: Each numbered step must be a separate bash tool call. Never combine multiple steps with &&, &, ;, or newlines. Do not run any other commands.""",
    },
    {
        "name": "chain_commands",
        "task": """You are in a Windows command shell. Perform exactly these steps:
1. Run `echo start && echo middle && echo end`.
2. Finish by running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
Do not run any other commands.""",
    },
]


def commands(message):
    return [action.get("command") for action in ((message.get("extra") or {}).get("actions") or [])]


def op_message(message):
    extra = message.get("extra") or {}
    role = message.get("role")
    if role == "assistant":
        return ("assistant", commands(message), message.get("content"))
    if role == "tool":
        return (
            "tool",
            extra.get("returncode"),
            extra.get("raw_output"),
            message.get("content"),
        )
    if role == "user":
        if commands(message):
            return ("user-action", commands(message))
        content = message.get("content")
        return ("user", content.rstrip() if isinstance(content, str) else content)
    if role == "system":
        return ("system", message.get("content"))
    if role == "exit":
        return ("exit", extra.get("exit_status"), extra.get("submission"))
    return (role, message.get("content"))


def normalize(path, workspace=None):
    trajectory = json.loads(Path(path).read_text(encoding="utf-8"))
    messages = trajectory["messages"]

    def scrub(value):
        if isinstance(value, str):
            if workspace is not None:
                for needle in (str(workspace), str(workspace).replace("\\", "\\\\")):
                    value = value.replace(needle, "<WORKSPACE>")
                return value
            return value
        if isinstance(value, list):

            return [scrub(item) for item in value]

        if isinstance(value, tuple):

            return tuple(scrub(item) for item in value)
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    return scrub(
        {
            "roles": [message.get("role") for message in messages],
            "commands": [commands(message) for message in messages if commands(message)],
            "observations": [
                op_message(message) for message in messages if message.get("role") == "tool"
            ],
            "exit": messages[-1].get("extra", {}),
            "normalized": [op_message(message) for message in messages],
        }
    )


def run_python(case, workspace, output_path):
    task_path = output_path.with_suffix(".task.txt")
    task_path.write_text(case["task"], encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_python_case.py"),
        str(task_path),
        str(workspace),
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    if completed.returncode != 0:
        raise RuntimeError(
            f"python runner failed for {case['name']}: {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    return output_path


def run_rust(case, workspace, output_path):
    command = [
        str(RUST_EXE),
        "-c",
        "mini.yaml",
        "-c",
        "model.model_kwargs.temperature=0",
        "-c",
        "model.model_kwargs.seed=1234",
        "-c",
        f"environment.cwd={workspace}",
        "--agent-class",
        "default",
        "-y",
        "--exit-immediately",
        "-m",
        "openai/deepseek-v4.1-flash",
        "-t",
        case["task"],
        "-o",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300)
    if completed.returncode != 0:
        raise RuntimeError(
            f"rust runner failed for {case['name']}: {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    return output_path


def main():
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACES.mkdir(parents=True, exist_ok=True)
    if not RUST_EXE.exists():
        raise SystemExit(f"missing rust binary: {RUST_EXE}")

    report = []
    all_ok = True
    selected = set(sys.argv[1:])
    cases = [case for case in CASES if not selected or case["name"] in selected]
    for case in cases:
        py_workspace = WORKSPACES / f"{case['name']}_python"
        rust_workspace = WORKSPACES / f"{case['name']}_rust"
        for workspace in (py_workspace, rust_workspace):
            if workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True)

        py_path = SUITE_DIR / f"{case['name']}.python.json"
        rust_path = SUITE_DIR / f"{case['name']}.rust.json"

        print(f"RUN python {case['name']}", flush=True)
        run_python(case, py_workspace, py_path)
        print(f"RUN rust   {case['name']}", flush=True)
        run_rust(case, rust_workspace, rust_path)

        py = normalize(py_path, py_workspace)
        rs = normalize(rust_path, rust_workspace)
        exact = py["normalized"] == rs["normalized"]
        roles_match = py["roles"] == rs["roles"]
        action_match = py["commands"] == rs["commands"]
        observation_match = py["observations"] == rs["observations"]
        exit_match = py["exit"] == rs["exit"]
        assistant_match = [
            message.get("content")
            for message in json.loads(py_path.read_text(encoding="utf-8"))["messages"]
            if message.get("role") == "assistant"
        ] == [
            message.get("content")
            for message in json.loads(rust_path.read_text(encoding="utf-8"))["messages"]
            if message.get("role") == "assistant"
        ]
        case_ok = roles_match and action_match and observation_match and exit_match
        all_ok = all_ok and case_ok
        report.append(
            {
                "name": case["name"],
                "exact_with_assistant_content": exact,
                "role_match": roles_match,
                "action_match": action_match,
                "observation_match": observation_match,
                "exit_match": exit_match,
                "assistant_content_match": assistant_match,
                "python_commands": py["commands"],
                "rust_commands": rs["commands"],
                "python_exit": py["exit"],
                "rust_exit": rs["exit"],
            }
        )
        print(
            f"{case['name']}: exact={exact} actions={action_match} "
            f"observations={observation_match} exit={exit_match} assistant={assistant_match}"
        )
        if not case_ok:
            print("  python commands:", py["commands"])
            print("  rust   commands:", rs["commands"])
            print("  python observations:", py["observations"])
            print("  rust   observations:", rs["observations"])

    report_path = SUITE_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nALL_OK={all_ok}")
    print(f"REPORT={report_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())











