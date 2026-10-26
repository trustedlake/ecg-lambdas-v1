"""v1 adapter: a lead with under 1 s validated falls back to the traced line, flagged as uncertain."""
import types

import numpy as np

import lambda_v1


def _export(analysis, amplitude):
    rec = {'name': 'II', 'row': 1, 'column': 0, 'box': [0, 0, 1, 1], 'baseline_y_px': 5.0}
    result = types.SimpleNamespace(leads=[rec])
    arrays = {'II__analysis_mv': analysis, 'II__amplitude_mv': amplitude}
    return lambda_v1._lead_export(result, arrays, {'x': {'pixels_per_mm': 4}}, {'speed_mm_s': 25}, '3x4+1')


def test_validated_lead_is_sent_as_analysis():
    trace = np.sin(np.arange(400) / 10)
    _, leads = _export(trace, trace)
    assert leads[0]['signal_source'] == 'analysis' and leads[0]['quality'] == 'good'


def test_blank_analysis_falls_back_to_traced_line():
    trace = np.sin(np.arange(400) / 10)
    _, leads = _export(np.full(400, np.nan), trace)
    lead = leads[0]
    assert lead['signal_source'] == 'traced_unvalidated' and lead['quality'] == 'uncertain'
    assert sum(v is not None for v in lead['samples']) == 400


def test_no_traced_line_drops_the_lead():
    _, leads = _export(np.full(400, np.nan), np.full(400, np.nan))
    assert leads == []
