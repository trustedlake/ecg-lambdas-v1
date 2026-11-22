"""Batch version of run_local.py: before (template rows) vs after (automatic rows), baseline_peaks only.

    run_batch.py --in-dir /w/in50 --out /w/batch50
"""
import argparse, io, json, os, sys, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lambda_v1
from research_methods import registry

registry.METHODS = {'baseline_peaks': registry.METHODS['baseline_peaks']}
real_detect = lambda_v1._detect_layout


def run(image, out, auto):
    os.environ['RESULTS_BUCKET'] = 'local'
    os.environ['ECG_METHOD_WORKERS'] = '1'
    body = image.read_bytes()

    class FakeS3:
        def head_object(self, Bucket, Key):
            return {'ContentLength': len(body)}

        def get_object(self, Bucket, Key):
            return {'Body': io.BytesIO(body)}

    def put(bucket, key, data, mime):
        f = out/key
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data if isinstance(data, bytes) else data.encode())

    lambda_v1.s3 = FakeS3()
    lambda_v1._put = put
    lambda_v1._detect_layout = real_detect if auto else (lambda *a: {'applied': False, 'reason': 'disabled_for_before'})
    event = {'request_id': str(uuid.uuid4()), 'layout': '3x4+1', 'contract_version': '2.0',
             'calibration': {'speed_mm_s': 25, 'gain_mm_mv': 10}, 'image': {'bucket': 'local', 'key': image.name}}
    response = lambda_v1.handler(event, None)
    out.mkdir(parents=True, exist_ok=True)
    (out/'response.json').write_text(json.dumps(response, indent=1, ensure_ascii=False))
    return response


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    for item in json.loads((a.in_dir/'index.json').read_text()):
        for mode in ('before', 'after'):
            dest = a.out/item['id']/mode
            if (dest/'response.json').exists():
                continue
            try:
                r = run(a.in_dir/item['file'], dest, mode == 'after')
                print(item['id'], mode, r.get('status'), flush=True)
            except Exception as exc:
                print(item['id'], mode, 'EXC', repr(exc), flush=True)


if __name__ == '__main__':
    main()
