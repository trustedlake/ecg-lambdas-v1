# ecg-analyser Lambda (analyze + refine)
Handler `lambda_handler.py` + `Dockerfile.lambda`. Build context: copy `ecg/`, `review/`, `configs/`, `requirements.txt` from `ecg-analyser/extractor/` next to these two files, then `docker build -f Dockerfile.lambda`.
