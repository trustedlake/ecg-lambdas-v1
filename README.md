# ecg-lambdas-v1

Production ECG reading, approved Selector and interpretation components used by [trustedlake/ecg-backend](https://github.com/trustedlake/ecg-backend). The reader preserves gaps and uncertainty in the extracted signal; it does not invent a trace or issue a clinical diagnosis.

| Component | Location |
| --- | --- |
| Lambda request, response and error contract | [`LAMBDA_CONTRACT.md`](LAMBDA_CONTRACT.md) |
| Real reader: six independent OpenCV methods, image Lambda and approved Selector | [`lambdas/real_reading/`](lambdas/real_reading/) |
| Interpretation Lambda, shared contract, tests and Terraform | [`lambdas/`](lambdas/README.md) |

## Production

AWS account `785236377332`, region `us-east-1`. A push to `main` triggers the `ecg-lambdas` CodePipeline through the authorized `ecg-github-frontend` connection.

The pipeline tests the source, builds and publishes the real `ecg-reading` container and the `ecg-interpretation` package, then runs a smoke test. On smoke-test failure it restores the previous Lambda code. The Selector artifact is stored outside Git and its checksum must match the approved model `cb0f395367ec` before reading begins.

## Local checks

```bash
pip install pillow boto3
python -m unittest discover -s lambdas/tests -v
```

For deployment and recovery, see [the backend deployment guide](https://github.com/trustedlake/ecg-backend/blob/main/docs/deployment.md).
