#!/usr/bin/env bash
# Builds the ZIP package of the ecg-interpretation lambda in mock_lambdas/build/interpretation/.
# (ecg-reading is a container image: see real_reading/assemble.sh.)
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build
mkdir -p build/interpretation
cp ../interpretation/handler.py ../interpretation/diagnoser.py build/interpretation/
cp -R ecg_contract build/interpretation/
find build -name '__pycache__' -prune -exec rm -rf {} +
echo "ok: $(du -sh build/interpretation | cut -f1) interpretation"
