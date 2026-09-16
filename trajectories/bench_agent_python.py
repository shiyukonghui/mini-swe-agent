import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
from minisweagent.agents.default import DefaultAgent


class FastModel:
    def __init__(self, steps):
        self.steps = steps
        self.calls = 0
        self.config = {}

    def query(self, messages, **kwargs):
        self.calls += 1
        if self.calls >= self.steps:
            return {
                "role": "exit",
                "content": "",
                "extra": {"exit_status": "Submitted", "submission": "", "cost": 0.0},
            }
        return {
            "role": "assistant",
            "content": "x",
            "extra": {"actions": [{"command": "echo x"}], "cost": 0.0},
        }

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [{"role": "user", "content": "observed"} for _ in outputs]

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class FastEnv:
    def __init__(self):
        self.config = {}

    def execute(self, action, cwd="", **kwargs):
        return {"output": "ok", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
agent = DefaultAgent(
    FastModel(steps),
    FastEnv(),
    system_template="system",
    instance_template="{{task}}",
    step_limit=0,
    cost_limit=0.0,
)
start = time.perf_counter()
agent.run("bench")
elapsed = time.perf_counter() - start
print({"steps": steps, "messages": len(agent.messages), "seconds": elapsed, "steps_per_second": steps / elapsed if elapsed else None})
