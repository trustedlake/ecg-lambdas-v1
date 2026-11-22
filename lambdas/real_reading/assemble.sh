#!/usr/bin/env bash
# Assembles real_reading/build/ (Docker context) from the vendored algorithm/ + ecg_contract.
# The selector model (.pkl, too big for git) is read from models/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build && mkdir -p build/app/selector
cp Dockerfile requirements.txt build/
for d in ecg research_methods review configs; do cp -R "algorithm/$d" build/app/; done
cp src/lambda_v1.py src/reading_mock.py build/app/
cp algorithm/selector/{score.py,features.py,data.py} build/app/selector/
cp models/model_selector.pkl build/app/selector/
cp -R ../ecg_contract build/app/
find build -name '__pycache__' -prune -exec rm -rf {} +
sha256sum build/app/selector/model_selector.pkl | cut -c1-12
