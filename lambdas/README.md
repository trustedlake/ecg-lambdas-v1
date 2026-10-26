# Production Lambdas

The production stack implements [`../LAMBDA_CONTRACT.md`](../LAMBDA_CONTRACT.md) for two functions.

| AWS function | Package | Production behavior |
| --- | --- | --- |
| `ecg-reading` | `real_reading/` container image | Extracts signals with six independent OpenCV methods, preserves gaps, and scores valid candidates only with the approved Selector model. |
| `ecg-interpretation` | `interpretation/` ZIP | Validates the extracted signal and returns automated measurements and findings. It is a screening aid and does not replace clinical review. |

`ecg_contract/` supplies the shared envelope, validation and error catalog. A request may include the documented `mock` scenario only for controlled test paths; it is not the normal production-reading path.

## Tests

```bash
pip install pillow boto3
python -m unittest discover -s lambdas/tests -v
python lambdas/smoke_aws.py
```

## CI/CD

A push to `main` in `trustedlake/ecg-lambdas-v1` through the `ecg-github-frontend` CodeStar connection starts CodePipeline `ecg-lambdas` in `us-east-1`.

1. Test and Build run in parallel. Build creates the real `ecg-reading` container image in ECR and the interpretation ZIP. It downloads the Selector model from the versioned model bucket and verifies its checksum before building the image.
2. Deploy publishes both functions, records the commit as `CODE_VERSION`, then runs `smoke_aws.py`.
3. If the smoke test fails, Deploy restores the previous image, ZIP and code version.

Terraform owns the surrounding infrastructure and ignores function code and `CODE_VERSION`, so a Terraform apply does not overwrite a pipeline release. The approved Selector model and its checksum are release controls; do not replace either as part of a normal deployment.

## Test scenarios

`ecg-reading` and `ecg-interpretation` accept the documented `mock` field for controlled scenarios. The normal reading path omits this field and runs the real extractor. See [`../LAMBDA_CONTRACT.md`](../LAMBDA_CONTRACT.md) for request and response details.
