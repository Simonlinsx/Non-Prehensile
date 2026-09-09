#!/usr/bin/env python3
"""Reverse a later patch in a temporary copy, then verify an earlier patch."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('later', type=Path)
parser.add_argument('earlier', type=Path)
args = parser.parse_args()
paths = set()
for patch in (args.later, args.earlier):
    for line in patch.read_text().splitlines():
        if line.startswith('+++ b/'):
            path = Path(line.removeprefix('+++ b/'))
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe patch path')
            paths.add(path)
with tempfile.TemporaryDirectory(prefix='push_anything_patch_check_') as tmp:
    for path in paths:
        dest = Path(tmp) / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.root / path, dest)
    subprocess.run(['git', 'apply', '--reverse', str(args.later.resolve())], cwd=tmp, check=True)
    subprocess.run(['git', 'apply', '--reverse', '--check', str(args.earlier.resolve())], cwd=tmp, check=True)
