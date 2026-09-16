import json, sys
from pathlib import Path
p=Path(sys.argv[1])
data=json.loads(p.read_text(encoding='utf-8'))
for i,m in enumerate(data['messages']):
    extra=m.get('extra') or {}
    print('---',i,m.get('role'))
    c=m.get('content')
    if c: print('content:',repr(c)[:1000])
    if extra.get('actions'): print('actions:',extra.get('actions'))
    if extra.get('interrupt_type'): print('interrupt_type:',extra.get('interrupt_type'))
    if m.get('role')=='exit': print('exit extra:',extra)
