#!/usr/bin/env bash
# Assembles real_reading/build/ (Docker context) from algorithm/ (the single algorithm source) + ecg_contract.
# The selector model (.pkl, too big for git) is read from models/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build && mkdir -p build/app/selector
cp Dockerfile requirements.txt build/
for d in ecg research_methods review configs; do cp -R "algorithm/$d" build/app/; done
cp src/lambda_v1.py src/reading_mock.py build/app/
cp algorithm/selector/{score.py,features.py,data.py} build/app/selector/
cp algorithm/source_info.py build/app/
cp models/model_selector.pkl build/app/selector/
cp -R ../ecg_contract build/app/
find build -name '__pycache__' -prune -exec rm -rf {} +
# The image must hold exactly the algorithm/ code that evaluation imports: same digest, recorded in the image.
PY="$(command -v python3 || command -v python)"
SRC_DIGEST="$(cd algorithm && "$PY" -c 'import source_info; print(source_info.digest())')"
APP_DIGEST="$(cd build/app && "$PY" -c 'import source_info; print(source_info.digest())')"
[ "$SRC_DIGEST" = "$APP_DIGEST" ] || { echo "algorithm digest mismatch: $SRC_DIGEST != $APP_DIGEST" >&2; exit 1; }
COMMIT="${CODEBUILD_RESOLVED_SOURCE_VERSION:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
printf '{"digest": "%s", "commit": "%s"}\n' "$APP_DIGEST" "$COMMIT" > build/app/ALGORITHM_SOURCE.json
echo "algorithm digest $APP_DIGEST commit $COMMIT"
sha256sum build/app/selector/model_selector.pkl | cut -c1-12
