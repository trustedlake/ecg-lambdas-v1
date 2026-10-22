"""Real handler calls against in-memory S3; no AWS account or clinical interpretation."""
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import lambda_handler as handler


class FakeS3:
    def __init__(self):
        self.objects = {}
    def head_object(self, Bucket, Key):
        return {'ContentLength': len(self.objects[Bucket, Key])}
    def download_file(self, bucket, key, destination):
        Path(destination).write_bytes(self.objects[bucket, key])
    def upload_file(self, source, bucket, key):
        self.objects[bucket, key] = Path(source).read_bytes()


def synthetic_process(source, out, config):
    assert config['calibration']['speed_mm_s'] == 50
    assert config['calibration']['gain_mm_mv'] == 5
    out.mkdir(parents=True)
    img = np.full((30, 40, 3), 255, np.uint8)
    img[12, 10:18] = 0
    Image.fromarray(img).save(out/'working_image.png')
    Image.fromarray(img).save(out/'visao_geral.png')
    mask=np.zeros((12,15),bool)
    mask[5,2:5]=True
    np.savez_compressed(out/'trace_masks.npz',II=mask)
    values={'x_px':np.arange(15,dtype='float32'),'y_px':np.full(15,np.nan,dtype='float32'),
       'y_min_px':np.full(15,np.nan,dtype='float32'),'y_max_px':np.full(15,np.nan,dtype='float32'),
       'observed':np.zeros(15,bool),'ambiguous':np.zeros(15,bool),
       'selected_valid':np.zeros(15,bool),'valid':np.zeros(15,bool),
       'analysis_mv':np.full(15,np.nan,dtype='float32'),
       'amplitude_mv':np.full(15,np.nan,dtype='float32'),
       'amplitude_min_mv':np.full(15,np.nan,dtype='float32'),
       'amplitude_max_mv':np.full(15,np.nan,dtype='float32')}
    values['observed'][2:5]=True
    values['valid'][2:5]=True
    np.savez_compressed(out/'signals.npz',**{'II__'+k:v for k,v in values.items()})
    meta={'status':'CONSISTENCY_PASS','warnings':[],
           'leads':[{'name':'II','mask_origin':[8,7],'coverage':.2,'valid_samples':3}]}
    (out/'result.json').write_text(json.dumps(meta),encoding='utf8')
    return meta


class HandlerIntegrationTests(unittest.TestCase):
    def test_analyze_refine_twice_and_keep_previous_artifacts(self):
        attempt=str(uuid.uuid4())
        s3=FakeS3()
        buffer=io.BytesIO(); Image.new('RGB',(4,4),'white').save(buffer,format='PNG')
        s3.objects['images','exam/input.png']=buffer.getvalue()
        evt={'operation':'analyze','attempt_id':attempt,'images_bucket':'images','results_bucket':'out',
             'imagem_s3_key':'exam/input.png','calibration':{'speed_mm_s':50,'gain_mm_mv':5}}
        with patch.object(handler.boto3,'client',return_value=s3),patch.object(handler,'process',side_effect=synthetic_process):
            first=handler.handler(evt,None)
            self.assertEqual(first['status'],'CONSISTENCY_PASS')
            self.assertIn('working_image',first['artifacts'])
            prev=first['artifacts']
            for x in (14,16):
                strokes=[{'lead':'II','mode':'include','points':[{'x':x/39,'y':12/29}]}]
                evt={'operation':'refine','attempt_id':attempt,'results_bucket':'out',
                     'lead':'II','strokes':strokes,'parent_artifacts':prev}
                result=handler.handler(evt,None)
                self.assertEqual(result['status'],'MANUAL_REVIEW_REQUIRED')
                self.assertIn('correction_preview',result['artifacts'])
                self.assertNotEqual(prev['signals'],result['artifacts']['signals'])
                self.assertIn(('out',prev['signals']),s3.objects)
                prev={**result['artifacts'],'visao_geral':result['artifacts']['correction_preview']}
            self.assertEqual(len([x for x in s3.objects if x[0]=='out' and x[1].endswith('signals.npz')]),3)

    def test_config_requires_explicit_calibration(self):
        with self.assertRaisesRegex(ValueError,'Explicit'):
            handler._analysis_config({})
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            handler._analysis_config({'calibration':{'speed_mm_s':200,'gain_mm_mv':10}})
        profile=handler._analysis_config({'calibration':{'speed_mm_s':25,'gain_mm_mv':10}})
        self.assertEqual(profile['calibration']['speed_mm_s'],25)
        self.assertEqual(profile['calibration']['gain_mm_mv'],10)

    def test_unsupported_extension_rejected_before_upload(self):
        s3=FakeS3(); aid=str(uuid.uuid4())
        e={'operation':'analyze','attempt_id':aid,'images_bucket':'images','results_bucket':'out',
           'imagem_s3_key':'x/image.gif','calibration':{'speed_mm_s':25,'gain_mm_mv':10}}
        with patch.object(handler.boto3,'client',return_value=s3):
            with self.assertRaisesRegex(ValueError,'Unsupported image format'):
                handler.handler(e,None)


class LambdaSecurityTests(unittest.TestCase):
    def test_malformed_paint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bad=[{'lead':'II','mode':'include','points':[{'x':float('nan'),'y':.3}]}]
            with self.assertRaisesRegex(ValueError,'normalized'):
                handler._paint(bad,'II',(40,30),Path(directory))
            with self.assertRaisesRegex(ValueError,'Invalid lead'):
                handler._paint([{'lead':'V1','mode':'exclude','points':[{'x':.1,'y':.2}]}], 'II',(40,30),Path(directory))
    def test_oversized_image_rejected_before_processing(self):
        aid=str(uuid.uuid4());s3=FakeS3()
        s3.objects['images','exam/input.png']=b'x'*(handler.MAX_INPUT_BYTES+1)
        event={'operation':'analyze','attempt_id':aid,'results_bucket':'out','images_bucket':'images',
               'imagem_s3_key':'exam/input.png','calibration':{'speed_mm_s':25,'gain_mm_mv':10}}
        with patch.object(handler.boto3,'client',return_value=s3),patch.object(handler,'process') as process:
            with self.assertRaisesRegex(ValueError,'maximum upload'):
                handler.handler(event,None)
            process.assert_not_called()
