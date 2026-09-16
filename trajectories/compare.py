import json
from pathlib import Path

py = json.loads(Path("trajectories/python_simple.json").read_text(encoding="utf-8"))
rs = json.loads(Path("trajectories/rust_simple.json").read_text(encoding="utf-8"))

def actions(msg):
    return [a.get("command") for a in (msg.get("extra") or {}).get("actions", [])]

def obs(msg):
    extra = msg.get("extra") or {}
    return {
        "raw_output": extra.get("raw_output"),
        "returncode": extra.get("returncode"),
        "exception_info": extra.get("exception_info"),
    }

print("python messages", len(py["messages"]))
print("rust   messages", len(rs["messages"]))
maxn = max(len(py["messages"]), len(rs["messages"]))
ok = True
for i in range(maxn):
    p = py["messages"][i] if i < len(py["messages"]) else {}
    r = rs["messages"][i] if i < len(rs["messages"]) else {}
    for key in ("role", "content"):
        if p.get(key) != r.get(key):
            ok = False
            print(f"[{i}] {key} differs")
            print("  py:", repr(p.get(key))[:300])
            print("  rs:", repr(r.get(key))[:300])
    if actions(p) != actions(r):
        ok = False
        print(f"[{i}] actions differ: {actions(p)!r} != {actions(r)!r}")
    if obs(p) != obs(r):
        ok = False
        print(f"[{i}] observation extras differ: {obs(p)!r} != {obs(r)!r}")
    if p.get("tool_call_id") != r.get("tool_call_id"):
        print(f"[{i}] tool_call_id differs (provider-specific): {p.get('tool_call_id')!r} != {r.get('tool_call_id')!r}")

last_p = py["messages"][-1].get("extra", {})
last_r = rs["messages"][-1].get("extra", {})
for key in ("exit_status", "submission"):
    if last_p.get(key) != last_r.get(key):
        ok = False
        print(f"exit {key} differs: {last_p.get(key)!r} != {last_r.get(key)!r}")

print("OPERATION_TRAJECTORY_MATCH =", ok)
