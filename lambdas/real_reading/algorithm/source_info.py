"""Identity of the algorithm code: one digest for evaluation results and the production image.

digest() hashes every file of ecg/, research_methods/, review/, configs/ and
selector/{score,features,data}.py (paths relative to this folder, CRLF -> LF, no
__pycache__), so the same code gives the same digest on Windows and in the Lambda image.
python source_info.py [root]  -> prints the JSON identity.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGES = ('ecg', 'research_methods', 'review', 'configs')
SELECTOR = ('score.py', 'features.py', 'data.py')


def files(root=HERE):
    root = Path(root)
    out = [p for d in PACKAGES for p in (root / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    out += [root / 'selector' / f for f in SELECTOR]
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def digest(root=HERE):
    root = Path(root)
    h = hashlib.sha256()
    for p in files(root):
        h.update(p.relative_to(root).as_posix().encode() + b'\0')
        h.update(p.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(['git', '-C', str(HERE), *args], capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def identity(root=HERE):
    """digest + commit. commit comes from git, else from $ALGORITHM_COMMIT (CI build)."""
    commit = _git('rev-parse', 'HEAD') or os.environ.get('ALGORITHM_COMMIT')
    status = _git('status', '--porcelain', '--', '.')
    return {'algorithm_dir': str(Path(root)), 'digest': digest(root), 'commit': commit,
            'dirty': bool(status) if status is not None else None}


if __name__ == '__main__':
    print(json.dumps(identity(sys.argv[1] if len(sys.argv) > 1 else HERE), indent=2))
