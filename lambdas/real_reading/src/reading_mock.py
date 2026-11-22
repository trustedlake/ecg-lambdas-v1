"""Mock mode of the reading Lambda (contract 2.0), selected by the `mock` field.

Same Lambda, same input validation and same response shape as the real run;
only the signal is synthetic. Nothing here touches the extraction methods.
"""
import math
import uuid

from ecg_contract import (ContractError, READ_FAILURE_CODES, READ_REFUSAL_CODES, SCENARIO_FAILURE,
                          SCENARIO_REFUSED, SCENARIO_SUCCESS, SCENARIO_WARNINGS, LAYOUTS,
                          resolve_mock, warning)

SCENARIOS = (SCENARIO_SUCCESS, SCENARIO_WARNINGS, SCENARIO_FAILURE, SCENARIO_REFUSED)
SCENARIO_CODES = {SCENARIO_FAILURE: READ_FAILURE_CODES, SCENARIO_REFUSED: READ_REFUSAL_CODES}
METHOD_ROWS = (('baseline_peaks', 'peaks34-323b4e6', 70.0), ('chromatic_ink', 'neutral-color-v1', 76.0),
               ('faint_ink', 'faint-color-v1', 81.0), ('uneven_photo', 'local-rgb-v1', 80.0),
               ('grayscale_contrast', 'local-luminance-v1', 79.0), ('same_appearance', 'mock-v1', 26.0))
RATE_HZ = 200.0
LEAD_SCALE = {'I': .8, 'II': 1.2, 'III': .5, 'aVR': -.9, 'aVL': .4, 'aVF': .8,
              'V1': .9, 'V2': 1.4, 'V3': 1.5, 'V4': 1.7, 'V5': 1.6, 'V6': 1.3}


def wants_mock(event):
    return isinstance(event, dict) and event.get('mock') is not None


def _samples(scale, seconds):
    out = []
    for i in range(int(seconds * RATE_HZ)):
        phase = (i / RATE_HZ) % (60 / 72)
        qrs = math.exp(-((phase - .2) / .012) ** 2)
        t_wave = .25 * math.exp(-((phase - .45) / .05) ** 2)
        out.append(round(scale * (qrs + t_wave), 4))
    return out


def _leads(layout):
    rows, rhythm = LAYOUTS[layout]
    leads = []
    for r, row in enumerate(rows):
        seconds = 10.0 / len(row)
        for c, name in enumerate(row):
            leads.append({'name': name, 'kind': 'panel', 'row': r, 'col': c,
                          'start_s': round(c * seconds, 4), 'duration_s': seconds,
                          'coverage': 1.0, 'quality': 'good', 'samples': _samples(LEAD_SCALE[name], seconds)})
    for i, name in enumerate(rhythm):
        leads.append({'name': name, 'kind': 'rhythm', 'row': len(rows) + i, 'col': 0, 'start_s': 0.0,
                      'duration_s': 10.0, 'coverage': 1.0, 'quality': 'good',
                      'samples': _samples(LEAD_SCALE[name], 10.0)})
    return leads


def mock_read(event, env, layout, calibration, source):
    from lambda_v1 import _file, _get_image, _json
    scenario, code = resolve_mock(event, None, SCENARIOS, SCENARIO_CODES)
    env.scenario = scenario
    if scenario == SCENARIO_FAILURE:
        raise ContractError(code or 'READ_TRACE_EXTRACTION_FAILED')
    if scenario == SCENARIO_REFUSED:
        raise ContractError(code or 'READ_REFUSED_LOW_CONTRAST')
    raw, suffix = _get_image(source)
    prefix = f'reading/{env.request_id}/{uuid.uuid4().hex}/mock/'
    mime = {'.jpg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}[suffix]
    image = _file(prefix + 'preview' + suffix, raw, mime)
    leads = _file(prefix + 'leads.json', _json({'leads': _leads(layout)}), 'application/json')
    best = max(METHOD_ROWS, key=lambda r: r[2])[0]
    rows = []
    for name, version, score in METHOD_ROWS:
        flagged = scenario == SCENARIO_WARNINGS and name == best
        rows.append({'method': name, 'version': version, 'status': 'OK', 'selector_score': score,
                     'sampling_rate_hz': RATE_HZ, 'reason': None,
                     'warnings': [warning('AMBIGUOUS_TRACE', lead='V1')] if flagged else [],
                     'files': {'json': _file(prefix + name + '.json', _json({'method': name, 'mock': True}),
                                             'application/json'),
                               'image': image, 'leads': leads}})
    result = {'layout': layout, 'calibration': calibration, 'units': 'mV', 'methods': rows}
    return env.success(result, [True] if scenario == SCENARIO_WARNINGS else None)
