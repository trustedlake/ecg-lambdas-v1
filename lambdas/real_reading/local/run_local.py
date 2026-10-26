"""Run the reading Lambda handler locally, inside its own Docker image, with no AWS access.

The S3 read and writes are replaced by local files: the input image comes from --image and every
object the handler would upload is written under --out/<key>. Same code path as production.

    docker run --rm -v <build/app>:/var/task -v <images>:/in -v <out>:/out --entrypoint python \
        <image> /var/task/local/run_local.py --image /in/x.png --out /out
"""
import argparse, json, os, sys, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lambda_v1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--layout', default='3x4+1')
    a = p.parse_args()
    os.environ['RESULTS_BUCKET'] = 'local'
    os.environ['ECG_METHOD_WORKERS'] = '1'
    body = a.image.read_bytes()

    class FakeS3:
        def head_object(self, Bucket, Key):
            return {'ContentLength': len(body)}

        def get_object(self, Bucket, Key):
            import io
            return {'Body': io.BytesIO(body)}

    def put(bucket, key, data, mime):
        f = a.out/key
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data if isinstance(data, bytes) else data.encode())

    lambda_v1.s3 = FakeS3()
    lambda_v1._put = put
    event = {'request_id': str(uuid.uuid4()), 'layout': a.layout, 'contract_version': '2.0',
             'calibration': {'speed_mm_s': 25, 'gain_mm_mv': 10},
             'image': {'bucket': 'local', 'key': a.image.name}}
    response = lambda_v1.handler(event, None)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'response.json').write_text(json.dumps(response, indent=1, ensure_ascii=False))
    print(json.dumps({'status': response.get('status'),
                      'methods': [(m['method'], m['status'], m.get('selector_score'))
                                  for m in (response.get('result') or {}).get('methods', [])]}))


if __name__ == '__main__':
    main()
