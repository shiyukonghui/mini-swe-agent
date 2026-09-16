import sys
from pathlib import Path
sys.path.insert(0, str(Path('trajectories').resolve()))
import parity_suite as ps
py = ps.normalize(Path('trajectories/suite/error_recovery.python.json'), Path('trajectories/suite/workspaces/error_recovery_python'))
rs = ps.normalize(Path('trajectories/suite/error_recovery.rust.json'), Path('trajectories/suite/workspaces/error_recovery_rust'))
for i,(a,b) in enumerate(zip(py['observations'], rs['observations'])):
    if a != b:
        print('diff',i)
        for j,(x,y) in enumerate(zip(a,b)):
            if x != y:
                print(' field',j)
                print('  py:',repr(x))
                print('  rs:',repr(y))
