> **Update (reading contract 2.0):** `reading/handler.py` references the real integration `ecg-analyser/extractor/lambda_v1.py`. The existing ZIP build deliberately keeps the legacy mock in `reading/mock_handler.py` and does NOT publish the real reading. Interpretation is still a mock (contract 1.0). See `../LAMBDA_CONTRACT.md` and `ecg-analyser/extractor/LAMBDA_V1_INTEGRATION.md` before any deploy.

# ECG reading and interpretation mock lambdas

They implement the **v1 contract** ([`../LAMBDA_CONTRACT.md`](../LAMBDA_CONTRACT.md)) with synthetic results,
to validate the app → backend → lambdas flow before the real algorithms (OpenCV 5).

| AWS function | Code | What it really does | What is synthetic |
|---|---|---|---|
| `ecg-reading` | `reading/mock_handler.py` | validates the payload, reads and decodes the photo from S3, draws the grid and trace over it, writes `overlay.png` and `signal.json` | the signal (72 bpm) and the grid position |
| `ecg-interpretation` | `../interpretation/handler.py` | validates sex and signal, measures HR and amplitudes on the received signal, applies sex-specific thresholds | PR, QRS, QT and axis |

`ecg_contract/` (envelope, error/warning catalog, layouts) is copied into both packages and can be reused by the real lambdas.

## Scenarios

`"mock": {"scenario": "success | success_with_warnings | failure | refused", "code": "<optional>"}` in the payload.
Without that field, the function's `MOCK_SCENARIO` env applies (Terraform variable `mock_scenario`) and, failing that, `success`.
The full table is in section 4 of the contract.

## Tests

```bash
pip install pillow boto3
python -m unittest discover -s lambdas/tests -v   # offline, fake S3
python lambdas/smoke_aws.py                       # against the deployed functions (profile trustedlake)
```

## Deploy (no Docker)

```bash
lambdas/build.sh                                   # builds build/{reading,interpretation}/ (Pillow arm64)
cd lambdas/infra
terraform init
terraform plan  -var="code_version=$(git rev-parse --short HEAD)" -out=tfplan
terraform apply tfplan
```

Independent stack in us-east-1 (state `s3://ecg-terraform-state-785236377332-us-east-1/ecg-mock-lambdas/terraform.tfstate`). It only
**reads** the `ecg-lambdas` buckets and changes nothing there. Runtime python3.12 arm64, CloudWatch logs with
30-day retention. The backend's IAM principal needs `lambda:InvokeFunction` on the ARNs in `terraform output function_arns`.

To switch to the real lambdas, point `AWS_LAMBDA_ECG_READING` / `AWS_LAMBDA_ECG_INTERPRETATION` in the backend to the new functions.

## CI/CD (pipeline `ecg-lambdas`, us-east-1)

A push to `main` of `tl-gmiler/ecg-lambdas-v1` (CodeStar connection `ecg-github`) runs CodePipeline `ecg-lambdas`.
It deploys straight to production, with no manual approval.

1. **Test** (`ecg-lambdas-test`) and **Build** (`ecg-lambdas-build`) run in parallel. Build creates the `ecg-analyser:<sha7>`
   image in ECR (no local Docker) and the interpretation zip. The model `model_selector.pkl` is downloaded from
   `s3://ecg-lambdas-models-785236377332-us-east-1` and checked against `lambdas/real_reading/model.sha256`.
2. **Deploy** (`ecg-lambdas-deploy`, `ci/deploy.sh`) runs `update-function-code` on `ecg-reading` and `ecg-interpretation`,
   sets `CODE_VERSION` to the commit, then runs `smoke_aws.py`.
3. **Rollback:** if the smoke test fails, the previous image, zip and `CODE_VERSION` are restored and the stage fails.

Failures publish to the SNS topic `ecg-alerts` (EventBridge rule `ecg-lambdas-pipeline-failed`).
Terraform ignores the function code and `CODE_VERSION` (`ignore_changes`), so `apply` never overwrites what the pipeline
deployed. The ECR repo `ecg-analyser` keeps the last 15 images.

**Manual:** the models bucket `ecg-lambdas-models-785236377332-us-east-1` (versioned) was created by hand. To change the
model, upload the new file there and update `model.sha256`.

## The `mock` field

There are no separate mock Lambdas. `ecg-reading` and `ecg-interpretation` accept the field
`mock: {"scenario": "success" | "success_with_warnings" | "failure" | "refused"}` in the payload
(interpretation: no `refused`). With it, reading really validates the input and returns a
synthetic signal in the 2.0 format (6 methods); without it, it runs the real extraction.
