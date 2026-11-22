"""Offline contract-2 integration tests; no AWS, model or research dataset.

Run from sibling working tree containing ecg-analyser, or set
ECG_ANALYSER_EXTRACTOR_PATH to its extractor directory.
"""
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve()
V1 = HERE.parents[1]
EXTRACTOR = Path(os.environ.get('ECG_ANALYSER_EXTRACTOR_PATH') or
                 HERE.parents[3] / 'ecg-analyser' / 'extractor')
sys.path.insert(0, str(V1))
if EXTRACTOR.is_dir():
    sys.path[:0] = [str(EXTRACTOR), str(EXTRACTOR.parent)]
    try:
        from lambda_v1 import handler as real_handler
        import lambda_v1 as real
        from research_methods.contracts import Extraction, MethodAbstention
        from research_methods.registry import METHODS
    except ImportError:   # the reading code lives in real_reading/ now
        real = None
else:
    real = None

from test_handlers import FakeS3, _image_bytes

SIX = ['baseline_peaks', 'chromatic_ink', 'faint_ink', 'uneven_photo', 'grayscale_contrast', 'same_appearance']


@unittest.skipUnless(real is not None, 'ecg-analyser/extractor not available; set ECG_ANALYSER_EXTRACTOR_PATH')
class RealContractTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeS3()
        real.s3 = self.client
        os.environ['RESULTS_BUCKET'] = 'results-test'
        self.client.objects['images', 'ecg.png'] = _image_bytes('PNG', (1000, 700))
        self.request = {'contract_version': '2.0', 'request_id': str(uuid.uuid4()),
                        'image': {'bucket': 'images', 'key': 'ecg.png'}, 'layout': '3x4+1',
                        'calibration': {'speed_mm_s': 25, 'gain_mm_mv': 10}}

    def call(self, **updates):
        return real.handler({**self.request, **updates}, None)

    def test_calibration_and_layout_are_strict_before_s3(self):
        for changed, error in (
            ({'calibration': None}, 'CALIBRATION_INVALID'),
            ({'calibration': {'speed_mm_s': 25}}, 'CALIBRATION_INVALID'),
            ({'calibration': {'gain_mm_mv': 10}}, 'CALIBRATION_INVALID'),
            ({'calibration': {'speed_mm_s': True, 'gain_mm_mv': 10}}, 'CALIBRATION_INVALID'),
            ({'layout': None}, 'LAYOUT_MISSING'),
            ({'layout': '6x2'}, 'LAYOUT_UNSUPPORTED'),
            ({'contract_version': '1.0'}, 'UNSUPPORTED_CONTRACT_VERSION'),
            ({'image': {'bucket': 'images', 'key': 'ECG Images/1. Testes/a.png'}}, 'IMAGE_SOURCE_FORBIDDEN'),
        ):
            with self.subTest(changed=changed):
                response = self.call(**changed)
                self.assertEqual(response['status'], 'ERROR')
                self.assertEqual(response['error']['code'], error)
                self.assertEqual(response['contract_version'], '2.0')

    def test_missing_selector_model_fails_closed_before_accessing_s3(self):
        with patch.object(real, '_verify_selector_model', side_effect=FileNotFoundError('model_selector.pkl')):
            response = self.call()
        self.assertEqual(response['status'], 'ERROR')
        self.assertEqual(response['error']['code'], 'READ_SELECTOR_UNAVAILABLE')
        self.assertEqual(len(self.client.objects), 1)

    def test_six_method_envelope_and_refusal(self):
        image = np.full((700, 1000, 3), 235, np.uint8)
        geom = {'method': 'colored_grid_boundary', 'orientation': {'uncertain': False}}
        grid = {'valid': True, 'x': {'pixels_per_mm': 6}, 'y': {'pixels_per_mm': 6}}
        with patch.object(real, '_verify_selector_model'), \
             patch('research_methods.frozen_peaks34.ecg.preprocess.prepare', return_value=(image, geom)), \
             patch('research_methods.frozen_peaks34.ecg.grid.estimate_grid', return_value=grid),              patch.dict('research_methods.registry.METHODS', clear=True,
                        values={k: METHODS[k] for k in SIX}):
            def method_row(name, method, image, geo, gd, config, path):
                return ({'method': name, 'version': method.VERSION, 'status': 'OK' if name == 'baseline_peaks' else 'ABSTAINED',
                         'selector_score': 87.0 if name == 'baseline_peaks' else None,
                         'sampling_rate_hz': 150 if name == 'baseline_peaks' else None,
                         'reason': None, 'warnings': [], 'files': {'json': None, 'image': None, 'leads': None}}, [])
            with patch.object(real, '_method_run', side_effect=method_row):
                response = self.call()
                self.assertEqual(response['status'], 'OK_WITH_WARNINGS')
                self.assertEqual(len(response['result']['methods']), 6)
                self.assertEqual(SIX, [x['method'] for x in response['result']['methods']])
                self.assertNotIn('warnings', response)
            with patch.object(real, '_method_run', side_effect=lambda name, method, *args:
                   ({'method': name, 'status': 'BLOCKED', 'reason': 'gate', 'selector_score': None, 'files': {}}, [])):
                response = self.call()
                self.assertEqual(response['status'], 'REFUSED')
                self.assertEqual(response['error']['code'], 'READ_REFUSED_LOW_CONTRAST')
                self.assertEqual(len(response['error']['details']['methods']), 6)
            with patch.object(real, '_method_run', side_effect=lambda name, method, *args:
                   ({'method': name, 'status': 'ERROR', 'reason': 'SELECTOR_UNAVAILABLE', 'selector_score': None, 'files': {}}, [])):
                response = self.call()
                self.assertEqual(response['status'], 'ERROR')
                self.assertEqual(response['error']['code'], 'READ_SELECTOR_UNAVAILABLE')

    def test_mock_field_runs_the_same_lambda_with_synthetic_signal(self):
        for scenario, status in (('success', 'OK'), ('success_with_warnings', 'OK_WITH_WARNINGS')):
            with self.subTest(scenario=scenario):
                response = self.call(mock={'scenario': scenario})
                self.assertEqual(response['status'], status)
                self.assertNotIn('warnings', response)
                methods = response['result']['methods']
                self.assertEqual(len(methods), 6)
                self.assertTrue(all('warnings' in m for m in methods))
                leads = json.loads(self.client.objects[methods[0]['files']['leads']['bucket'],
                                                       methods[0]['files']['leads']['key']])
                self.assertEqual(len(leads['leads']), 13)
        response = self.call(mock={'scenario': 'refused'})
        self.assertEqual(response['status'], 'REFUSED')
        response = self.call(mock={'scenario': 'failure'})
        self.assertEqual(response['status'], 'ERROR')
        self.assertEqual(self.call(mock={'scenario': 'nope'})['error']['code'], 'MOCK_SCENARIO_INVALID')

    @staticmethod
    def measured_extraction():
        columns = 40
        mask = np.ones((8, columns), bool)
        trace = {'mask': mask, 'mask_origin': (10, 20),
                 'observed': np.ones(columns, bool), 'ambiguous': np.zeros(columns, bool),
                 'x_px': np.arange(10, 10 + columns, dtype=float),
                 'y_px': np.full(columns, 24.0),
                 'y_min_px': np.full(columns, 23.0), 'y_max_px': np.full(columns, 25.0),
                 'valid': np.array([False] * 5 + [True] * 35),
                 'baseline_y_px': 26.0, 'coverage': 1.0, 'unambiguous_fraction': .875}
        rec = {'name': 'II_longo', 'row': 3, 'column': 0, 'box': [10, 20, 50, 28],
               'baseline_y_px': 26.0, 'color': '#335577'}
        return Extraction([rec], [trace])

    def test_actual_convert_s3_and_gap_preservation(self):
        from selector import score
        image = np.full((100, 150, 3), 240, np.uint8)
        geom = {'method': 'colored_grid_boundary', 'orientation': {}}
        grid = {'valid': True, 'x': {'pixels_per_mm': 6.0}, 'y': {'pixels_per_mm': 6.0}}
        config = {'calibration': {'speed_mm_s': 25, 'gain_mm_mv': 10},
                  'requested_layout': '3x4+1', 'layout_label': 'test', 'review': {}}
        class Extractor:
            VERSION = 'test-only'
            @staticmethod
            def extract(ctx):
                return RealContractTests.measured_extraction()
        with patch('research_methods.frozen_peaks34.ecg.contrast_gate.assess_color_separation',
                   return_value={'block': False}), \
             patch.object(real, '_verify_selector_model'), \
             patch('selector.score.score_leads', return_value={'II': 88.5}), \
             patch('review.overlay.render', return_value=Image.new('RGB', (50, 50))):
            meta, warnings = real._method_run('baseline_peaks', Extractor, image, geom, grid, config, 'reading/test/run/')
        self.assertEqual(meta['status'], 'OK')
        self.assertEqual(meta['sampling_rate_hz'], 150.0)
        self.assertAlmostEqual(meta['selector_score'], 88.5)
        content = self.client.objects['results-test', meta['files']['leads']['key']]
        exported = json.loads(content)
        lead = exported['leads'][0]
        self.assertEqual((lead['name'], lead['kind']), ('II', 'rhythm'))
        self.assertEqual(lead['samples'][:5], [None] * 5)
        self.assertTrue(all(v is not None for v in lead['samples'][5:]))
        self.assertEqual(len(lead['samples']), 40)
        self.assertTrue(any(w['code'] == 'AMBIGUOUS_TRACE' for w in warnings) is False)
        self.assertIn('selector_per_lead', json.loads(self.client.objects['results-test', meta['files']['json']['key']]))

        with patch('research_methods.frozen_peaks34.ecg.contrast_gate.assess_color_separation',
                   return_value={'block': True, 'reason': 'red_grid_overlap'}):
            blocked, _ = real._method_run('chromatic_ink', Extractor, image, geom, grid, config, 'reading/test/run/')
        self.assertEqual(blocked['status'], 'BLOCKED')
        self.assertIsNone(blocked['files']['leads'])
        self.assertTrue(blocked['files']['image']['key'].endswith('rejection_preview.png'))

        with patch('research_methods.frozen_peaks34.ecg.contrast_gate.assess_color_separation',
                   return_value={'block': False}), \
             patch.object(real, '_verify_selector_model'), \
             patch('selector.score.score_leads', side_effect=FileNotFoundError('model_selector.pkl')):
            errored, _ = real._method_run('faint_ink', Extractor, image, geom, grid, config, 'reading/test/run/')
        self.assertEqual(errored['status'], 'ERROR')
        self.assertEqual(errored['reason'], 'SELECTOR_UNAVAILABLE')
        self.assertIsNone(errored['selector_score'])
        self.assertIsNone(errored['files']['leads'])

    def test_abstain_does_not_create_signals(self):
        class Refuse:
            VERSION = 'test'
            @staticmethod
            def extract(ctx):
                raise MethodAbstention('unavailable evidence')
        image = np.full((80, 150, 3), 255, np.uint8)
        metadata, _ = real._method_run('same_appearance', Refuse, image,
                                      {'method': 'full_image_fallback', 'orientation': {}},
                                      {'valid': False},
                                      {'calibration': {'speed_mm_s': 25, 'gain_mm_mv': 10}},
                                      'reading/test/run/')
        self.assertEqual(metadata['status'], 'ABSTAINED')
        self.assertIsNone(metadata['files']['leads'])
        self.assertTrue(metadata['files']['image']['key'].endswith('rejection_preview.png'))


if __name__ == '__main__':
    unittest.main()
