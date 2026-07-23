#!/usr/bin/env bash
# Re-point .venv at the interpreter in .python after this folder moves.
#
# The venv is self-contained: .venv/bin/python{,3,3.11} are real copied binaries
# (not symlinks) and .python/ holds the full interpreter, so nothing outside
# this directory is needed. The one path that cannot be made relative is
# `home` in .venv/pyvenv.cfg -- CPython resolves a relative `home` against the
# current working directory, which breaks. So if this folder is mounted at a
# different absolute path, run this script once:
#
#     bash relocate_venv.sh
#
# Idempotent; safe to run even if nothing moved.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$REPO/.venv/pyvenv.cfg"
PYDIR="$REPO/.python/cpython-3.11.15-linux-x86_64-gnu"

[[ -f "$CFG" ]] || { echo "error: no $CFG" >&2; exit 1; }
[[ -x "$PYDIR/bin/python3.11" ]] || { echo "error: no interpreter at $PYDIR" >&2; exit 1; }

current="$(sed -n 's/^home = //p' "$CFG")"
want="$PYDIR/bin"

if [[ "$current" == "$want" ]]; then
    echo "pyvenv.cfg already correct: $want"
else
    sed -i "s|^home = .*|home = $want|" "$CFG"
    echo "pyvenv.cfg home:"
    echo "  was: $current"
    echo "  now: $want"
fi

echo -n "check: "
"$REPO/.venv/bin/python" -c 'import sys; print("python", sys.version.split()[0], "OK ->", sys.prefix)'
