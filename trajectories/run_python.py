import json
from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

task = """Please perform exactly these steps:
1. Run the bash command `echo TRAJECTORY_STEP_1`.
2. After seeing its output, run `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
Do not run any other commands.
"""

cfg = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text(encoding="utf-8"))
model_cfg = dict(cfg["model"])
model_cfg["model_kwargs"] = {**model_cfg.get("model_kwargs", {}), "temperature": 0}
model_cfg["cost_tracking"] = "ignore_errors"
model = LitellmModel(model_name="openai/deepseek-v4.1-flash", **model_cfg)
env = LocalEnvironment(**cfg["environment"])
task = task.rstrip("\n")
agent = DefaultAgent(model, env, **cfg["agent"])
agent.run(task)
out = Path("trajectories/python_simple.json")
agent.save(out)
print(json.dumps({"n_messages": len(agent.messages), "exit": agent.messages[-1].get("extra", {})}))

