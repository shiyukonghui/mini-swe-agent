import json
from pathlib import Path
py=json.loads(Path('trajectories/python_simple.json').read_text(encoding='utf-8'))
rs=json.loads(Path('trajectories/rust_simple.json').read_text(encoding='utf-8'))
p=py['messages'][1]['content']; r=rs['messages'][1]['content']
print(len(p), len(r))
for i,(a,b) in enumerate(zip(p,r)):
    if a!=b:
        print('diff at',i)
        print('py:',repr(p[max(0,i-80):i+120]))
        print('rs:',repr(r[max(0,i-80):i+120]))
        break
else:
    print('prefix equal; tail diff')
    print('py tail:',repr(p[-200:]))
    print('rs tail:',repr(r[-200:]))
