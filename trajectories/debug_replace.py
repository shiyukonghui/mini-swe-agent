import json
from pathlib import Path
p=Path(r'D:\Rust\mini-swe-agent\trajectories\suite\error_recovery.python.json')
ws=Path(r'D:\Rust\mini-swe-agent\trajectories\suite\workspaces\error_recovery_python')
data=json.loads(p.read_text(encoding='utf-8'))
print('ws=',repr(str(ws)))
for i,m in enumerate(data['messages']):
    extra=m.get('extra') or {}
    raw=extra.get('raw_output','')
    print(i, m.get('role'), repr(raw)[:200], 'contains=', str(ws) in raw)
    if str(ws) in raw:
        print('replace=',repr(raw.replace(str(ws),'<WORKSPACE>')))
