import json
from pathlib import Path

py = json.loads(Path("trajectories/python_simple.json").read_text(encoding="utf-8"))
rs = json.loads(Path("trajectories/rust_simple.json").read_text(encoding="utf-8"))

def commands(msg):
    return [a.get("command") for a in ((msg.get("extra") or {}).get("actions") or [])]

def op_message(msg):
    extra = msg.get("extra") or {}
    role = msg.get("role")
    if role == "assistant":
        return ("assistant", commands(msg))
    if role == "tool":
        return ("tool", extra.get("returncode"), extra.get("raw_output"), msg.get("content"))
    if role == "user":
        content = msg.get("content")
        if (extra.get("actions")):
            return ("user-action", commands(msg))
        return ("user", content.rstrip() if isinstance(content, str) else content)
    if role == "system":
        return ("system", msg.get("content"))
    if role == "exit":
        return ("exit", extra.get("exit_status"), extra.get("submission"))
    return (role, msg.get("content"))

py_ops = [op_message(m) for m in py["messages"]]
rs_ops = [op_message(m) for m in rs["messages"]]

print("python roles:", [m.get("role") for m in py["messages"]])
print("rust   roles:", [m.get("role") for m in rs["messages"]])
print("python commands:", [commands(m) for m in py["messages"] if commands(m)])
print("rust   commands:", [commands(m) for m in rs["messages"] if commands(m)])
print("python observations:", [(m.get("extra") or {}).get("returncode") for m in py["messages"] if m.get("role") == "tool"])
print("rust   observations:", [(m.get("extra") or {}).get("returncode") for m in rs["messages"] if m.get("role") == "tool"])
print("python exit:", py["messages"][-1].get("extra", {}))
print("rust   exit:", rs["messages"][-1].get("extra", {}))
print("ROLE_SEQUENCE_MATCH =", [m.get("role") for m in py["messages"]] == [m.get("role") for m in rs["messages"]])
print("ACTION_SEQUENCE_MATCH =", [commands(m) for m in py["messages"]] == [commands(m) for m in rs["messages"]])
print("OBSERVATION_SEQUENCE_MATCH =", [op_message(m) for m in py["messages"] if m.get("role") in ("tool",)] == [op_message(m) for m in rs["messages"] if m.get("role") in ("tool",)])
print("EXIT_MATCH =", py["messages"][-1].get("extra", {}) == rs["messages"][-1].get("extra", {}))
print("NORMALIZED_OPERATION_TRAJECTORY_MATCH =", py_ops == rs_ops)
if py_ops != rs_ops:
    for i, (p, r) in enumerate(zip(py_ops, rs_ops)):
        if p != r:
            print("first difference at", i)
            print("py:", repr(p)[:500])
            print("rs:", repr(r)[:500])
            break
