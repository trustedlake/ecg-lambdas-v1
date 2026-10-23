#!/usr/bin/env bash
# Runs in the Deploy action (ci/buildspec-deploy.yml). Inputs: $CODEBUILD_SRC_DIR_image/{image.env,interpretation.zip}.
set -euo pipefail

# shellcheck disable=SC1091
source "$CODEBUILD_SRC_DIR_image/image.env"
NEW_IMAGE="$ECR_REPOSITORY_URL:$IMAGE_TAG"
NEW_ZIP="$CODEBUILD_SRC_DIR_image/interpretation.zip"

PREV_IMAGE="$(aws lambda get-function --function-name "$FUNCTION_READING" --query 'Code.ImageUri' --output text)"
curl -fsSL "$(aws lambda get-function --function-name "$FUNCTION_INTERPRETATION" --query 'Code.Location' --output text)" -o /tmp/prev.zip
echo "reading: $PREV_IMAGE -> $NEW_IMAGE"

# Sets CODE_VERSION (meta.engine_version) keeping the other variables.
set_version() { # $1 function, $2 version
  local vars
  vars="$(aws lambda get-function-configuration --function-name "$1" --query 'Environment.Variables' --output json | python3 -c 'import json,sys; v=json.load(sys.stdin); v["CODE_VERSION"]=sys.argv[1]; print(json.dumps({"Variables": v}))' "$2")"
  aws lambda update-function-configuration --function-name "$1" --environment "$vars" >/dev/null
  wait_updated "$1"
}
version_of() { aws lambda get-function-configuration --function-name "$1" --query 'Environment.Variables.CODE_VERSION' --output text; }

wait_updated() { aws lambda wait function-updated-v2 --function-name "$1"; }

PREV_VERSION_READING="$(version_of "$FUNCTION_READING")"
PREV_VERSION_INTERPRETATION="$(version_of "$FUNCTION_INTERPRETATION")"

deploy() { # $1 image, $2 zip, $3 reading version, $4 interpretation version
  aws lambda update-function-code --function-name "$FUNCTION_READING" --image-uri "$1" >/dev/null
  aws lambda update-function-code --function-name "$FUNCTION_INTERPRETATION" --zip-file "fileb://$2" >/dev/null
  wait_updated "$FUNCTION_READING"
  wait_updated "$FUNCTION_INTERPRETATION"
  set_version "$FUNCTION_READING" "$3"
  set_version "$FUNCTION_INTERPRETATION" "$4"
}

deploy "$NEW_IMAGE" "$NEW_ZIP" "$IMAGE_TAG" "$IMAGE_TAG"
if python mock_lambdas/smoke_aws.py --profile ""; then
  echo "Smoke test passed: $IMAGE_TAG is live."
else
  echo "Smoke test FAILED: rolling back to $PREV_IMAGE and the previous interpretation package." >&2
  deploy "$PREV_IMAGE" /tmp/prev.zip "$PREV_VERSION_READING" "$PREV_VERSION_INTERPRETATION"
  exit 1
fi
