import os
import sys
from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

task_path = Path(sys.argv[1])
workspace = Path(sys.argv[2])
output_path = Path(sys.argv[3])
task = task_path.read_text(encoding="utf-8")

cfg = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text(encoding="utf-8"))
model_cfg = dict(cfg["model"])
model_cfg["model_kwargs"] = {
    **model_cfg.get("model_kwargs", {}),
    "temperature": 0,
    "seed": 1234,
}
model_cfg["cost_tracking"] = "ignore_errors"
model = LitellmModel(model_name="openai/deepseek-v4.1-flash", **model_cfg)
env_cfg = dict(cfg["environment"])
env_cfg["cwd"] = str(workspace)
env = LocalEnvironment(**env_cfg)
agent = DefaultAgent(model, env, **cfg["agent"])
agent.config.output_path = output_path
agent.run(task)
agent.save(output_path)
