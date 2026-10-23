"""Adapter for ecg-lambdas-v1 reading contract 2.0.

Algorithm is vendored under algorithm/; the v1 reading handler is only an import shim.
No frozen research outputs, file-ID overrides or training data at runtime.
This module does NOT provision AWS resources or bypass the approved color gate.
"""
from copy import deepcopy
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from ecg_contract import (ContractError, Envelope, GAINS_MM_MV, LAYOUTS,
                          SPEEDS_MM_S, warning, validate_common)

LOG = logging.getLogger(__name__)
ENGINE = 'ecg-lambdas-v1'
ENGINE_VERSION = os.environ.get('CODE_VERSION', 'dev')
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
FORMAT_EXT = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}
# Only generic, layout-specific profiles; exclude D0/D1/... configurations
# based on image sets. Other v1 layouts have no verified generic config.
CONFIG_FOR_LAYOUT = {'3x4+1': 'report_normal.json', '12x1': 'data_12x1.json'}
s3 = None
_UPLOADS = None


def _s3():
    global s3
    if s3 is None:
        s3 = boto3.client('s3', config=Config(retries={'max_attempts': 3}))
    return s3


def _validate(event):
    request_id = validate_common(event, operation='read')
    layout = event.get('layout')
    if layout in (None, ''):
        raise ContractError('LAYOUT_MISSING')
    if layout not in LAYOUTS or layout not in CONFIG_FOR_LAYOUT:
        raise ContractError('LAYOUT_UNSUPPORTED', details={'received': layout, 'supported': sorted(CONFIG_FOR_LAYOUT)})
    calibration = event.get('calibration')
    if not isinstance(calibration, dict):
        raise ContractError('CALIBRATION_INVALID', details={'reason': 'calibration is required and must be explicit'})
    cal = {}
    for field, allowed in (('speed_mm_s', SPEEDS_MM_S), ('gain_mm_mv', GAINS_MM_MV)):
        value = calibration.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or float(value) not in allowed:
            raise ContractError('CALIBRATION_INVALID', details={'field': field, 'received': value, 'allowed': list(allowed)})
        cal[field] = float(value)
    source = event.get('image')
    if not isinstance(source, dict) or not all(isinstance(source.get(k), str) and source[k]
                                                    for k in ('bucket', 'key')):
        raise ContractError('IMAGE_MISSING')
    segments = source['key'].replace('\\', '/').lower().split('/')
    if any(segments[i:i+2] == ['ecg images', '1. testes'] for i in range(len(segments)-1)):
        raise ContractError('IMAGE_SOURCE_FORBIDDEN')
    return request_id, layout, cal, source


def _get_image(source):
    bucket, key = source['bucket'], source['key']
    try:
        client = _s3()
        head = client.head_object(Bucket=bucket, Key=key)
        length = head['ContentLength']
        if length > MAX_IMAGE_BYTES:
            raise ContractError('IMAGE_TOO_LARGE', details={'bytes': length, 'max_bytes': MAX_IMAGE_BYTES})
        body = client.get_object(Bucket=bucket, Key=key)['Body'].read(MAX_IMAGE_BYTES + 1)
    except ClientError as e:
        error = e.response.get('Error', {}).get('Code')
        if error in ('404', '403', 'NoSuchKey', 'NotFound', 'AccessDenied'):
            raise ContractError('IMAGE_NOT_FOUND') from e
        raise ContractError('STORAGE_ERROR') from e
    except BotoCoreError as e:
        raise ContractError('STORAGE_ERROR') from e
    if len(body) > MAX_IMAGE_BYTES:
        raise ContractError('IMAGE_TOO_LARGE')
    try:
        with Image.open(io.BytesIO(body)) as original:
            fmt = original.format
            if fmt not in FORMAT_EXT:
                raise ContractError('IMAGE_FORMAT_UNSUPPORTED', details={'detected': fmt})
            # Protect before decode against zip bombs and tiny inputs.
            w, h = original.size
            if w * h > MAX_IMAGE_PIXELS:
                raise ContractError('IMAGE_TOO_LARGE', details={'reason': 'pixel_limit'})
            rgb = ImageOps.exif_transpose(original).convert('RGB')
            w, h = rgb.size
            if min(w, h) < 500 or max(w, h) < 800:
                raise ContractError('IMAGE_TOO_SMALL', details={'width': w, 'height': h})
    except ContractError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise ContractError('IMAGE_UNREADABLE') from e
    return body, FORMAT_EXT[fmt]


def _config(layout, calibration):
    from ecg.config import load_config, resolve
    source = Path(__file__).resolve().parent / 'configs' / CONFIG_FOR_LAYOUT[layout]
    config = load_config(source)
    config['calibration'] = {
        'speed_mm_s': calibration['speed_mm_s'],
        'gain_mm_mv': calibration['gain_mm_mv'],
        'speed_source': 'Explicit request v2',
        'gain_source': 'Explicit request v2',
    }
    # Filenames do not pick layouts or image-specific refinements.
    return resolve(config, 'ecg_input.png')


def _file(key, body, mime):
    bucket = os.environ.get('RESULTS_BUCKET')
    if not bucket:
        raise ContractError('STORAGE_ERROR', details={'reason': 'RESULTS_BUCKET not set'})
    if _UPLOADS is not None:  # method subprocess: the parent performs the upload
        _UPLOADS.append((bucket, key, body, mime))
    else:
        _put(bucket, key, body, mime)
    return {'bucket': bucket, 'key': key, 'content_type': mime}


def _put(bucket, key, body, mime):
    try:
        _s3().put_object(Bucket=bucket, Key=key, Body=body, ContentType=mime)
    except (ClientError, BotoCoreError) as e:
        raise ContractError('STORAGE_ERROR') from e


def _json(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')


def _safe_json(x):
    """Convert numpy scalars/arrays and all nonfinite values to JSON null."""
    import numpy as np
    if isinstance(x, np.ndarray):
        return _safe_json(x.tolist())
    if isinstance(x, np.generic):
        return _safe_json(x.item())
    if isinstance(x, (float, int)) and not isinstance(x, bool) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe_json(v) for v in x]
    return x


def _png(image):
    b = io.BytesIO()
    image.save(b, format='PNG')
    return b.getvalue()


def _rejection_image(image):
    preview = Image.fromarray(image).convert('RGB')
    d = ImageDraw.Draw(preview)
    d.rectangle((0, 0, preview.width-1, preview.height-1), outline=(220, 35, 35), width=6)
    return preview


def _px_list(values):
    """Photo pixel position of every sample (gaps as null), so findings can be located on the photo."""
    import numpy as np
    if values is None:
        return None
    return _safe_json(np.round(np.asarray(values, dtype=float), 1).tolist())


def _lead_export(result, arrays, grid, cal, layout):
    """Native pixel sampling, gaps as null, rhythm II_longo -> II/rhythm."""
    import numpy as np
    fs = float(grid['x']['pixels_per_mm'] * cal['speed_mm_s'])
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError('Invalid grid-derived native sample rate')
    leads = []
    columns = len(LAYOUTS[layout][0][0])
    for record in result.leads:
        original_name = record['name']
        values = arrays[original_name + '__analysis_mv']
        samples = _safe_json(np.asarray(values, dtype=float).tolist())
        uncertain = False
        if sum(v is not None for v in samples) < fs:
            # Under 1 s marked valid: fall back to the traced line, flagged so interpretation treats it as uncertain.
            traced = _safe_json(np.asarray(arrays[original_name + '__amplitude_mv'], dtype=float).tolist())
            if sum(v is not None for v in traced) >= fs:
                samples, uncertain = traced, True
        if not samples or all(v is None for v in samples):
            continue
        rhythm = original_name.endswith('_longo')
        name = original_name[:-6] if rhythm else original_name
        row = int(record['row'])
        col = int(record.get('column', 0))
        leads.append({
            'name': name, 'kind': 'rhythm' if rhythm else 'panel',
            'row': row, 'col': col,
            'start_s': 0.0 if rhythm else round(col * 10.0/columns, 4),
            'duration_s': round(len(samples)/fs, 4),
            'coverage': round(sum(v is not None for v in samples)/len(samples), 4),
            'quality': 'uncertain' if uncertain else 'good' if sum(v is not None for v in samples)/len(samples) >= .90 else 'partial',
            'signal_source': 'traced_unvalidated' if uncertain else 'analysis',
            'bbox_px': _safe_json(record['box']),
            'baseline_y_px': float(record['baseline_y_px']),
            'x_px': _px_list(arrays.get(original_name + '__x_px')),
            'y_px': _px_list(arrays.get(original_name + '__y_px')),
            'samples': samples,
        })
    return fs, leads


def _warnings(geometry, grid, traces, names):
    out = []
    if geometry.get('orientation', {}).get('uncertain'):
        out.append(warning('ORIENTATION_UNCERTAIN'))
    if geometry['method'].endswith('fallback'):
        out.append(warning('GEOMETRY_FALLBACK'))
    if not grid['valid']:
        out.append(warning('GRID_CALIBRATION_UNCERTAIN'))
    for name, trace in zip(names, traces):
        label = name[:-6] if name.endswith('_longo') else name
        if float(trace['coverage']) < .90:
            out.append(warning('LEAD_PARTIAL_TRACE', lead=label))
    return out


def _verify_selector_model():
    """Fail closed unless the approved, immutable selector artifact is present."""
    import selector.score as approved_score
    model = Path(approved_score.__file__).resolve().with_name('model_selector.pkl')
    if not model.is_file():
        raise FileNotFoundError('SELECTOR_UNAVAILABLE: model_selector.pkl not included in the package')
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if not digest.startswith('cb0f395367ec'):
        raise RuntimeError('SELECTOR_UNAVAILABLE: model_selector.pkl hash fora do aprovado')


def _method_run(name, method, image, geometry, grid, config, base):
    import numpy as np
    from research_methods.frozen_peaks34.ecg.extract import convert
    from research_methods.contracts import Context, MethodAbstention
    from research_methods.frozen_peaks34.ecg.contrast_gate import assess_color_separation
    from review.overlay import render, overlay_geometry

    metadata = {'method': name, 'version': method.VERSION, 'status': 'ERROR', 'selector_score': None,
                'sampling_rate_hz': None, 'reason': None, 'warnings': [],
                'files': {'json': None, 'image': None, 'leads': None}}
    source_row = {'method': name, 'version': method.VERSION}
    output_image = None
    signal = None
    warnings = []
    try:
        result = method.extract(Context(image, deepcopy(config), deepcopy(geometry), deepcopy(grid)))
        integrity = result.validate(image.shape)
        gate = result.diagnostics.get('color_gate')
        if gate is None:
            gate = assess_color_separation(image, result.leads, result.traces, grid)
        source_row.update({'integrity': integrity, 'color_gate': _safe_json(gate),
                           'geometry': _safe_json(geometry), 'grid': _safe_json(grid),
                           'diagnostics': _safe_json(result.diagnostics),
                           'leads': [{**_safe_json(record),
                                      'coverage': float(trace['coverage']),
                                      'unambiguous_fraction': float(trace['unambiguous_fraction'])}
                                     for record, trace in zip(result.leads, result.traces)]})
        if gate['block']:
            metadata['status'] = 'BLOCKED'
            metadata['reason'] = str(gate.get('reason') or 'color_gate.block')
            output_image = ('rejection_preview.png', _rejection_image(image))
        elif not grid['valid']:
            metadata['status'] = 'ABSTAINED'
            metadata['reason'] = 'GRID_CALIBRATION_UNCERTAIN: invalid grid; physical signal unavailable'
            output_image = ('rejection_preview.png', _rejection_image(image))
        else:
            arrays = {}
            for lead, trace in zip(result.leads, result.traces):
                values, calibrated = convert(trace, grid, config['calibration'], config.get('review'), lead['name'])
                if not calibrated:
                    raise ValueError('Uncalibrated signal rejected')
                arrays.update({lead['name']+'__'+k: v for k, v in values.items()})
            rate, leads = _lead_export(result, arrays, grid, config['calibration'], config['requested_layout'])
            if not leads:
                metadata['status'] = 'ABSTAINED'
                metadata['reason'] = 'No measured nonempty lead'
                output_image = ('rejection_preview.png', _rejection_image(image))
            else:
                # Use exactly the approved selector's load_leads+score_leads+record_score;
                # never substitute appearance-router, fallback scores or retraining.
                _verify_selector_model()
                from selector.score import load_leads, score_leads, record_score
                b = io.BytesIO()
                np.savez_compressed(b, **arrays)
                b.seek(0)
                per_lead = score_leads(load_leads(b))
                if not per_lead:
                    raise RuntimeError('Approved selector has no scoreable leads')
                score = float(record_score(per_lead))
                if not math.isfinite(score) or not 0 <= score <= 100:
                    raise RuntimeError('Invalid selector score')
                metadata['status'] = 'OK'
                metadata['selector_score'] = round(score, 3)
                metadata['sampling_rate_hz'] = round(rate, 5)
                signal = {'sampling_rate_hz': rate, 'units': 'mV', 'leads': leads,
                          'image_size_px': [int(image.shape[1]), int(image.shape[0])],
                          'overlay_offset_px': overlay_geometry(image.shape[1], image.shape[0])[0],
                          'overlay_size_px': overlay_geometry(image.shape[1], image.shape[0])[1],
                          'layout': config['requested_layout'],
                          'calibration': deepcopy(config['calibration'])}
                warnings = _warnings(geometry, grid, result.traces, [x['name'] for x in result.leads])
                output_image = ('preview.png', render(image, result.leads, result.traces,
                                grid, config, 'RESEARCH ONLY - UNVALIDATED'))
                source_row['selector_per_lead'] = _safe_json(per_lead)
    except MethodAbstention as exc:
        metadata['status'] = 'ABSTAINED'
        metadata['reason'] = str(exc)
        output_image = ('rejection_preview.png', _rejection_image(image))
    except Exception as exc:
        LOG.exception('Method %s failed', name)
        metadata['status'] = 'ERROR'
        metadata['reason'] = ('SELECTOR_UNAVAILABLE' if 'SELECTOR_UNAVAILABLE' in str(exc)
                              or (isinstance(exc, (FileNotFoundError, ImportError))
                                  and ('model_selector.pkl' in str(exc) or 'selector' in str(exc)))
                              else 'METHOD_PROCESSING_ERROR')
        source_row['error_type'] = type(exc).__name__
        metadata['selector_score'] = None
        metadata['sampling_rate_hz'] = None
        signal = None
        output_image = None
        warnings = []
        # No signal is published and no score is invented after any failure.
    metadata['warnings'] = _distinct_warnings(warnings)
    source_row.update({'status': metadata['status'], 'reason': metadata['reason'],
                       'selector_score': metadata['selector_score'],
                       'sampling_rate_hz': metadata['sampling_rate_hz'],
                       'signals_exported': signal is not None})
    # Always store result.json; blocked/abstained also have a rejection preview.
    prefix = base + name + '/'
    metadata['files']['json'] = _file(prefix+'result.json', _json(_safe_json(source_row)), 'application/json')
    if output_image:
        metadata['files']['image'] = _file(prefix+output_image[0], _png(output_image[1]), 'image/png')
    if signal:
        metadata['files']['leads'] = _file(prefix+'leads.json', _json(_safe_json(signal)), 'application/json')
    return metadata, warnings


def _method_child(conn, name, method, image, geometry, grid, config, base):
    global _UPLOADS
    _UPLOADS = []
    try:
        row, w = _method_run(name, method, image, geometry, grid, config, base)
        conn.send(('ok', row, w, _UPLOADS))
    except BaseException as exc:
        conn.send(('error', repr(exc)))
    finally:
        conn.close()


def _run_methods(methods, image, geometry, grid, config, base):
    """Run the methods in forked subprocesses (pure-Python hot spots hold the GIL, so threads
    do not help). Lambda has no /dev/shm, so plain Process+Pipe is used, not Pool/Queue.
    Falls back to sequential when fork is unavailable or ECG_METHOD_WORKERS=1."""
    import multiprocessing
    workers = max(1, int(os.environ.get('ECG_METHOD_WORKERS', '6')))
    rows, warnings = [], []
    if workers == 1 or 'fork' not in multiprocessing.get_all_start_methods():
        for name, method in methods.items():
            row, w = _method_run(name, method, image, geometry, grid, config, base)
            rows.append(row)
            warnings.extend(w)
        return rows, warnings
    ctx = multiprocessing.get_context('fork')
    items = list(methods.items())
    for start in range(0, len(items), workers):
        running = []
        for name, method in items[start:start+workers]:
            parent, child = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_method_child,
                               args=(child, name, method, image, geometry, grid, config, base))
            proc.start()
            child.close()
            running.append((name, proc, parent))
        for name, proc, parent in running:
            try:
                message = parent.recv()
            except EOFError:
                message = ('error', 'method subprocess died without a result (OOM?)')
            proc.join()
            if message[0] != 'ok':
                raise RuntimeError(f'method {name} subprocess failed: {message[1]}')
            _, row, w, uploads = message
            for bucket, key, body, mime in uploads:
                _put(bucket, key, body, mime)
            rows.append(row)
            warnings.extend(w)
    return rows, warnings


def handler(event, context):
    env = Envelope('read', ENGINE, ENGINE_VERSION)
    try:
        env.request_id, layout, calibration, source = _validate(event)
        import reading_mock
        if reading_mock.wants_mock(event):
            response = reading_mock.mock_read(event, env, layout, calibration, source)
            response.pop('warnings', None)
            return response
        # Do not spend Lambda time or create artifacts with an absent/altered model.
        try:
            _verify_selector_model()
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            raise ContractError('READ_SELECTOR_UNAVAILABLE') from exc
        raw, suffix = _get_image(source)
        config = _config(layout, calibration)
        config['requested_layout'] = layout
        import cv2
        from research_methods.frozen_peaks34.ecg.preprocess import prepare
        from research_methods.frozen_peaks34.ecg.grid import estimate_grid
        from research_methods.registry import METHODS
        cv2.setNumThreads(1)
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            path = Path(directory)/('input'+suffix)
            path.write_bytes(raw)
            image, geometry = prepare(path, config)
            grid = estimate_grid(image, config)
            image.setflags(write=False)
            prefix = f'reading/{env.request_id}/{uuid.uuid4().hex}/'
            rows, _ = _run_methods(METHODS, image, geometry, grid, config, prefix)
        result = {'layout': layout, 'calibration': calibration, 'units': 'mV', 'methods': rows}
        ok = any(row['status'] == 'OK' for row in rows)
        all_refused = all(row['status'] in ('BLOCKED', 'ABSTAINED') for row in rows)
        if ok:
            # Warnings live inside each method; the envelope status only flags that some exist
            # (or that some method gave no signal, which must never be silently accepted).
            flagged = any(row['status'] != 'OK' or row['warnings'] for row in rows)
            response = env.success(result, [True] if flagged else None)
        elif all_refused:
            refusal = ('READ_REFUSED_LOW_CONTRAST' if any(row['status'] == 'BLOCKED' for row in rows)
                       else 'READ_REFUSED_UNREADABLE_TRACE')
            response = env.failure(ContractError(refusal, details={'methods': rows}))
        else:
            fail = ('READ_SELECTOR_UNAVAILABLE' if any(row['reason'] == 'SELECTOR_UNAVAILABLE' for row in rows)
                    else 'READ_TRACE_EXTRACTION_FAILED')
            response = env.failure(ContractError(fail, details={'methods': rows}))
    except ContractError as exc:
        LOG.info('read %s: %s', env.request_id, exc)
        response = env.failure(exc)
    except Exception:
        LOG.exception('read %s: unexpected error', env.request_id)
        response = env.failure(ContractError('INTERNAL_ERROR'))
    response.pop('warnings', None)
    return response


def _distinct_warnings(items):
    seen = set()
    out = []
    for item in items:
        key = (item['code'], item.get('lead'))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
