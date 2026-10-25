"""v1 adapter: neither the colour gate nor an unconfirmed grid refuses a reading; both only flag it."""
from unittest.mock import patch

import numpy as np
from PIL import Image

import lambda_v1
from research_methods.contracts import Extraction


class _Extractor:
    VERSION = 'test-only'

    @staticmethod
    def extract(ctx):
        columns = 400
        trace = {'mask': np.ones((8, columns), bool), 'mask_origin': (10, 20),
                 'observed': np.ones(columns, bool), 'ambiguous': np.zeros(columns, bool),
                 'x_px': np.arange(10, 10 + columns, dtype=float), 'y_px': np.full(columns, 24.0),
                 'y_min_px': np.full(columns, 23.0), 'y_max_px': np.full(columns, 25.0),
                 'valid': np.ones(columns, bool), 'baseline_y_px': 26.0, 'coverage': 1.0,
                 'unambiguous_fraction': 1.0}
        rec = {'name': 'II_longo', 'row': 3, 'column': 0, 'box': [10, 20, 410, 28], 'baseline_y_px': 26.0}
        return Extraction([rec], [trace])


def _run(grid, gate):
    image = np.full((100, 450, 3), 240, np.uint8)
    config = {'calibration': {'speed_mm_s': 25, 'gain_mm_mv': 10}, 'requested_layout': '3x4+1', 'review': {}}
    with patch.dict('os.environ', {'RESULTS_BUCKET': 'results-test'}), \
         patch.object(lambda_v1, '_put'), \
         patch.object(lambda_v1, '_verify_selector_model'), \
         patch('research_methods.frozen_peaks34.ecg.contrast_gate.assess_color_separation', return_value=gate), \
         patch('selector.score.score_leads', return_value={'II': 70.0}), \
         patch('review.overlay.render', return_value=Image.new('RGB', (50, 50))):
        return lambda_v1._method_run('chromatic_ink', _Extractor, image, {'method': 'test', 'orientation': {}},
                                     grid, config, 'reading/test/run/')


GRID = {'valid': True, 'minor_square_mm': 1.0, 'x': {'pixels_per_mm': 6.0}, 'y': {'pixels_per_mm': 6.0}}


def test_colour_gate_flags_but_does_not_block():
    meta, warnings = _run(GRID, {'block': True, 'reason': 'ACHROMATIC_TRACE_GRID_NO_CHROMATIC_SEPARATION'})
    assert meta['status'] == 'OK' and meta['selector_score'] == 70.0
    assert meta['files']['leads'] is not None
    assert 'LOW_TRACE_GRID_CONTRAST' in [w['code'] for w in warnings]


def test_unconfirmed_grid_is_used_and_flagged():
    image = np.zeros((100, 500, 3), np.uint8)
    grid = lambda_v1._usable_grid({'valid': False, 'minor_square_mm': 1.0,
                                   'x': {'spacing_px': 6.0, 'pixels_per_mm': 6.0, 'valid': False},
                                   'y': {'spacing_px': None, 'pixels_per_mm': None, 'valid': False}},
                                  image, {'grid_width_mm_hint': 250})
    assert grid['valid'] and not grid['measured_valid']
    assert grid['y']['pixels_per_mm'] == 6.0 and grid['y']['source'] == 'other_axis'
    meta, warnings = _run(grid, {'block': False})
    assert meta['status'] == 'OK'
    assert 'GRID_CALIBRATION_UNCERTAIN' in [w['code'] for w in warnings]


def test_grid_without_any_estimate_falls_back_to_page_width():
    image = np.zeros((100, 500, 3), np.uint8)
    empty = {'spacing_px': None, 'pixels_per_mm': None, 'valid': False}
    grid = lambda_v1._usable_grid({'valid': False, 'minor_square_mm': 1.0, 'x': dict(empty), 'y': dict(empty)},
                                  image, {'grid_width_mm_hint': 250})
    assert grid['x']['pixels_per_mm'] == grid['y']['pixels_per_mm'] == 2.0
    assert grid['x']['source'] == 'page_width_hint'


def test_confirmed_grid_is_untouched():
    assert lambda_v1._usable_grid(GRID, None, {}) is GRID
