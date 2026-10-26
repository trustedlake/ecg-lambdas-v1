# ECG reading adapter for `ecg-lambdas-v1` — contract 2.0

Entry point: `lambda_v1.handler(event, context)`, referenced by the v1 shim in `lambdas/reading/handler.py`.

This adapter lives **in this repository** (ecg-analyser), keeping v1 as the owner of the envelope/contract errors. There are no calls to frozen evaluation resources (`outputs/lote50_01/...`); six modules come directly from `research_methods.registry`. `baseline_peaks` and the utilities for `prepare`, `grid`, `convert` and `color_gate` use `frozen_peaks34`, whose manifest was verified by the tests. The classifier comes from `selector/score.py` without retraining.

## Flow

1. Full validation of `request_id`, version 2.0, layout, explicit speed/gain and bucket/key; `1. Testes` is forbidden.
2. Presence **and SHA-256 of the approved model** (`model_selector.pkl`, prefix `cb0f395367ec`) verified before downloading and processing the image; if missing/mismatched: `READ_SELECTOR_UNAVAILABLE`.
3. S3 `head_object/get_object` with limits on size, format and pixels. Decoding with EXIF preserved by `prepare`; the original image is not modified.
4. `prepare`, `estimate_grid` and six sequential `extract(Context(...))` calls (no parallelism, no dependencies on frozen outputs).
5. Candidate integrity via `Extraction.validate`; the chromatic gate only warns (`LOW_TRACE_GRID_CONTRAST`), it never blocks signal export. An unconfirmed grid is used with `GRID_CALIBRATION_UNCERTAIN` (missing axis: the other axis, else the page-width hint).
6. For each allowed candidate: `convert` the original into physical units; native signal, without resampling the output; score with **`load_leads`, `score_leads`, `record_score`** of the approved model, with no heuristic fallback.
7. `result.json`, `preview.png` (or `rejection_preview.png`) and `leads.json` under `reading/<request_id>/<run_id>/<method>/`, saved in `RESULTS_BUCKET`. Status and references are returned in `methods[]`.
8. Final 2.0 envelope, with `REFUSED` if all are blocked/abstained, `ERROR` if none produced a signal and there was an error.

## Layouts

`3x4+1` and `12x1` have static, generic profiles already available (`configs/report_normal.json` / `configs/data_12x1.json`), always with the calibration overridden **only** from the explicit input. Other v1 layouts have no matching generic configuration in the delivered materials and return `LAYOUT_UNSUPPORTED`. No fitting by file name or by image was done.

## Testing locally with the two repositories side by side

```bash
cd ecg-lambdas-v1
PYTHONPATH=lambdas python -m unittest discover -s lambdas/tests -v
cd ../ecg-analyser
PYTHONPATH=extractor python -m unittest discover -s extractor/tests -p 'test_research_methods.py' -v
PYTHONPATH=extractor python -m unittest discover -s extractor/tests -p 'test_selector.py' -v
```

In the first battery the integration tests discover `../ecg-analyser/extractor`. If it is elsewhere, set `ECG_ANALYSER_EXTRACTOR_PATH`. The `FakeS3` tests do not run AWS. The export test mocks **only the score** because the `model_selector.pkl` file did not come in the ZIP. The contract refuses real operation without that approved file.

## Deploy: not authorized, not performed

No container was published and no Terraform, IAM, timeout, memory or bucket was changed. The v1 `lambdas/build.sh` still builds the **legacy mock reading** package, from `reading/mock_handler.py`, to avoid unintentionally replacing an existing deploy. Do **not** use that build to publish the real adapter.

Once there is a cost/infra decision, package together `ecg_contract`, the `lambda_v1` module, the `ecg`, `research_methods`, `review` modules, `configs`, the `selector` folder + verified `model_selector.pkl` and compatible dependencies, and point the handler to `reading.handler.handler`. v1 currently declares 1769 MB/60 s; ECR, runtime, timeout and memory are pending decisions. `extractor/requirements.txt` in the ZIP still pins `opencv-python-headless==4.13.0.92`; compatibility with OpenCV 5 was not validated here, nor was the AWS environment.

## External dependencies to complete

- **Original approved model** (`selector/model_selector.pkl`, SHA `cb0f395367ec...`) — not included and not reconstructed.
- Images from `ECG Images/0. Treino` to test the six methods, measure times/memory and check the gate on samples 07/38. **Do not access `1. Testes`.**
- Django backend: select the method with the highest `selector_score` among `status == "OK"`, download `files.leads` and pass that JSON in the `signal` property of the `interpret` Lambda — not implemented in this pair of repositories.
- Decide on accepting the six methods still categorized as research, and the packaging/infra cost.

**Clinical use:** this is experimental signal extraction, with risk of inaccuracy. No `OK` attests medical validity or replaces clinical assessment.
