> **Integration pending deploy:** the real 2.0 read lives in `ecg-analyser/extractor/lambda_v1.py`; `lambdas/build.sh` still packages the old mock. See `LAMBDA_CONTRACT.md`.

# ecg-lambdas-v1

ECG **reading** and **interpretation** lambdas of the ECG project, and the v1 contract between them and the backend
([trustedlake/ecg-backend](https://github.com/trustedlake/ecg-backend)).

| What | Where |
|---|---|
| v1 contract: request, response, error catalog, scenarios and rules for the real lambdas | [`LAMBDA_CONTRACT.md`](LAMBDA_CONTRACT.md) |
| Deployed mocks (`ecg-reading`, `ecg-interpretation`), tests, smoke test and Terraform | [`lambdas/`](lambdas/README.md) |
| Shared module (envelope, validation, catalog), reusable by the real lambdas | [`lambdas/ecg_contract/`](lambdas/ecg_contract/__init__.py) |

AWS: account `785236377332`, region `us-east-1` (migrated from sa-east-1 on 2026-10-23; state in `ecg-terraform-state-785236377332-us-east-1`), profile `trustedlake`.

```bash
pip install pillow boto3
python -m unittest discover -s lambdas/tests -v
```

This repo is separate from [trustedlake/ecg-lambdas](https://github.com/trustedlake/ecg-lambdas), the old Docker pipeline
with a Step Function. This stack only reads the buckets from there.
