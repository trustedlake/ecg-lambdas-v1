#!/usr/bin/env bash
# Monta o pacote ZIP da lambda ecg-interpretation em mock_lambdas/build/interpretation/.
# (A ecg-reading é imagem de container: veja real_reading/assemble.sh.)
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build
mkdir -p build/interpretation
cp interpretation/handler.py build/interpretation/
cp -R ecg_contract build/interpretation/
find build -name '__pycache__' -prune -exec rm -rf {} +
echo "ok: $(du -sh build/interpretation | cut -f1) interpretation"
