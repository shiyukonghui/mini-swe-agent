import json
from pathlib import Path
suite=Path('trajectories/suite')
for case in ['file_roundtrip','python_compute','error_recovery','multi_file','chain_commands']:
    print('='*20, case)
    for lang in ['python','rust']:
        p=suite/f'{case}.{lang}.json'
        if not p.exists():
            print(lang, 'MISSING'); continue
        data=json.loads(p.read_text(encoding='utf-8'))
        msgs=data.get('messages', [])
        cmds=[]
        for m in msgs:
            for a in (m.get('extra') or {}).get('actions', []) or []:
                cmds.append(a.get('command'))
        exit_extra=msgs[-1].get('extra', {}) if msgs else {}
        print(lang, 'messages=',len(msgs), 'commands=',cmds, 'exit=',exit_extra)
