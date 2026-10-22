#!/usr/bin/env bash
# Assembles real_reading/build/ (Docker context) from the vendored algorithm/ + ecg_contract.
# Only the selector model (.pkl, too big for git) is read from ecg-analyser.
set -euo pipefail
cd "$(dirname "$0")"
A="${ECG_ANALYSER_DIR:-../../../ecg-analyser}"
rm -rf build && mkdir -p build/app/selector
cp Dockerfile requirements.txt build/
for d in ecg research_methods review configs; do cp -R "algorithm/$d" build/app/; done
cp src/lambda_v1.py src/reading_mock.py build/app/
cp algorithm/selector/{score.py,features.py,data.py} build/app/selector/
cp "$A/selector/model_selector.pkl" build/app/selector/
cp -R ../ecg_contract build/app/
find build -name '__pycache__' -prune -exec rm -rf {} +
sha256sum build/app/selector/model_selector.pkl | cut -c1-12
