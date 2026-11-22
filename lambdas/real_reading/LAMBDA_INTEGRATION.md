# AWS Lambda ECG Analyser — contract

| Operation | `status` | `artifacts` | S3 prefix |
|---|---|---|---|
| `analyze` | `CONSISTENCY_PASS` / `REVIEW_REQUIRED` | `result`, `signals`, `trace_masks`, `working_image`, `visao_geral` | `ecg-analyser/{attempt_id}/initial/` |
| `analyze` (blocked) | `REJECTED_LOW_TRACE_GRID_CONTRAST` | `result`, `rejection_preview` | `ecg-analyser/{attempt_id}/initial/` |
| `refine` | `MANUAL_REVIEW_REQUIRED` | `result`, `signals`, `trace_masks`, `working_image`, `correction_preview` (+ previous `visao_geral`) | `ecg-analyser/{attempt_id}/{uuid}/` |

The return also provides `warnings`, `grid`, `leads`, `manual_correction`. `refine` receives mask/strokes in `normalized_image_v1` coordinates: scale [0,1] over the already-rectified **working_image.png**; the handler rounds to `round(x*(width-1))`. Rejections do not accept painting, and corrected columns remain marked as ambiguous/invalid for clinical analysis.

Expected runtime: AWS Lambda container Python + NumPy + OpenCV-headless + Pillow + boto3. Suggested sizing **3 GB RAM, 240 s timeout, /tmp 2 GB**, subject to real testing; no deploy was done. The Docker image was built locally (1.25 GB) and the handler imports correctly. A synchronous Lambda takes up to 240 s plus cold start/network, and Django/Gunicorn/proxy need a compatible timeout.
