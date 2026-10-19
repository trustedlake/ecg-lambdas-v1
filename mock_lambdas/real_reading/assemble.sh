#!/usr/bin/env bash
# Assembles real_reading/build/ (Docker context) from ecg-analyser + ecg_contract.
set -euo pipefail
cd "$(dirname "$0")"
A="${ECG_ANALYSER_DIR:-../../../ecg-analyser}"
rm -rf build && mkdir -p build/app/selector
cp Dockerfile requirements.txt build/
for d in ecg research_methods review configs; do cp -R "$A/extractor/$d" build/app/; done
cp "$A/extractor/lambda_v1.py" "$A/extractor/reading_mock.py" build/app/
# Ship only the approved six methods: pin the registry to the committed version, so a method
# still under research in the working tree never reaches production.
git -C "$A" show HEAD:research_methods/registry.py > build/app/research_methods/registry.py
cp "$A/selector/"{score.py,features.py,data.py,model_selector.pkl} build/app/selector/
cp -R ../ecg_contract build/app/
find build -name '__pycache__' -prune -exec rm -rf {} +
sha256sum build/app/selector/model_selector.pkl | cut -c1-12
