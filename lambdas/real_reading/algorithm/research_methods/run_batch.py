"""Reprocess the fixed training manifest; production entrypoints are untouched."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback
import cv2
import numpy as np
from PIL import Image
from review.overlay import render
from .contracts import Context, MethodAbstention
from .registry import METHODS
from .features import measure
from .frozen_peaks34.ecg.config import resolve
from .frozen_peaks34.ecg.preprocess import prepare
from .frozen_peaks34.ecg.grid import estimate_grid
from .frozen_peaks34.ecg.extract import convert
from .frozen_peaks34.ecg.contrast_gate import assess_color_separation

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


def code_manifest():
    paths = sorted((ROOT/'research_methods').rglob('*.py')) + sorted((ROOT/'review').glob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def compare_npz(path, values):
    with np.load(path) as old:
        if set(old.files) != set(values):
            return False
        return all(np.array_equal(old[k], values[k], equal_nan=True) for k in old.files)


def mask_delta(old_root, current, shape):
    metadata = json.loads((old_root/'result.json').read_text(encoding='utf8'))
    if not (old_root/'trace_masks.npz').is_file():
        return {'previous_status': metadata['status'], 'comparable': False}
    added = removed = 0
    old_records = {lead['name']: lead for lead in metadata['leads']}
    with np.load(old_root/'trace_masks.npz') as masks:
        for lead, trace in zip(current.leads, current.traces):
            name = lead['name']
            if name not in masks or name not in old_records:
                return {'comparable': False, 'reason': 'Lead sets differ'}
            a = np.zeros(shape[:2], bool); b = a.copy()
            x,y = old_records[name]['mask_origin']; h,w = masks[name].shape
            if y+h > a.shape[0] or x+w > a.shape[1]:
                return {'comparable': False, 'reason': 'Working geometry differs'}
            a[y:y+h,x:x+w] = masks[name]
            x,y = trace['mask_origin']; h,w = trace['mask'].shape
            b[y:y+h,x:x+w] = trace['mask']
            added += int((b & ~a).sum()); removed += int((a & ~b).sum())
    return {'comparable': True, 'added_pixels': added, 'removed_pixels': removed,
            'previous_status': metadata['status'], 'note': 'Change counts, not accuracy.'}


def process_entry(job):
    entry, output, before = job
    cv2.setNumThreads(1)
    start = time.monotonic()
    ident = entry['id']; source = Path(entry['source'])
    baseline_root = ROOT/'outputs/lote50_01/peaks34/images'/ident
    baseline = json.loads((baseline_root/'result.json').read_text(encoding='utf8'))
    if '1. testes' in [p.lower() for p in source.parts]:
        raise ValueError('Held-out folder forbidden')
    if digest(source) != baseline['signature']['source_sha256']:
        raise ValueError('Source differs from frozen peaks34 input')
    config = resolve(baseline['configuration'], source)
    if config.get('trace_profile', 'standard') != 'standard':
        raise ValueError('Expected standard baseline')
    image, geometry = prepare(source, config, entry.get('page', 0))
    grid = estimate_grid(image, config)
    image.setflags(write=False)
    destination = Path(output)/'images'/ident
    destination.mkdir(parents=True, exist_ok=False)
    Image.fromarray(image).save(destination/'working.png')
    records = []
    for name, method in METHODS.items():
        target = destination/name; target.mkdir()
        began = time.monotonic()
        try:
            context = Context(image, deepcopy(config), deepcopy(geometry), deepcopy(grid))
            result = method.extract(context)
            integrity = result.validate(image.shape)
            gate = result.diagnostics.get('color_gate')
            if gate is None:
                gate = assess_color_separation(image, result.leads, result.traces, grid)
            masks = {lead['name']: trace['mask'] for lead, trace in zip(result.leads, result.traces)}
            arrays = {}
            # Research overlays may show blocked candidates; no signals are exported for them.
            if not gate['block']:
                for lead, trace in zip(result.leads, result.traces):
                    values, _ = convert(trace, grid, config['calibration'], config.get('review'), lead['name'])
                    arrays.update({lead['name']+'__'+key: value for key,value in values.items()})
                np.savez_compressed(target/'signals.npz', **arrays)
            np.savez_compressed(target/'candidate_masks.npz', **masks)
            blocked = bool(gate['block'])
            row = {'method': name, 'version': method.VERSION, 'status': 'RESEARCH_ONLY',
                   'production_blocked': blocked, 'color_gate': gate, 'integrity': integrity,
                   'geometry': geometry, 'grid': grid,
                   'leads': [{**lead, 'mask_origin': trace['mask_origin'],
                              'coverage': trace['coverage'], 'unambiguous_fraction': trace['unambiguous_fraction']}
                             for lead,trace in zip(result.leads,result.traces)],
                   'diagnostics': result.diagnostics,
                   'delta_from_grid41': mask_delta(Path(before)/'images'/ident, result, image.shape),
                   'mean_coverage': float(np.mean([t['coverage'] for t in result.traces])),
                   'mean_unambiguous': float(np.mean([t['unambiguous_fraction'] for t in result.traces])),
                   'signals_exported': bool(arrays), 'elapsed_s': time.monotonic()-began}
            if name == 'baseline_peaks':
                old_blocked = baseline['status'].startswith('REJECTED_')
                row['baseline_equivalence'] = {'block_equal': blocked == old_blocked,
                    'mask_equal': None if blocked or old_blocked else compare_npz(baseline_root/'trace_masks.npz', masks),
                    'signal_equal': None if blocked or old_blocked else compare_npz(baseline_root/'signals.npz', arrays)}
                write(destination/'features.json', measure(image, result.leads))
            render(image, result.leads, result.traces, grid, config,
                   'RESEARCH ONLY - BLOCKED' if blocked else 'RESEARCH ONLY - UNVALIDATED').save(target/'preview.png')
            write(target/'result.json', row)
        except MethodAbstention as error:
            row = {'method': name, 'version': method.VERSION, 'status': 'ABSTAINED',
                   'reason': str(error), 'signals_exported': False,
                   'elapsed_s': time.monotonic()-began}
            Image.fromarray(image).save(target/'rejection_preview.png')
            write(target/'result.json', row)
        except Exception as error:
            row = {'method': name, 'version': method.VERSION, 'status': 'ERROR',
                   'error': str(error), 'traceback': traceback.format_exc()}
            write(target/'result.json', row)
        records.append(row)
    summary = {'id': ident, 'source_sha256': digest(source), 'methods': records,
               'elapsed_s': time.monotonic()-start}
    write(destination/'result.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--ids', nargs='*', help='Pilot only; omit for all 50')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    selection = json.loads((ROOT/'outputs/lote50_01/selection.json').read_text(encoding='utf8'))
    records = selection['records']
    if args.ids:
        if set(args.ids)-{r['id'] for r in records}:
            parser.error('Unknown pilot ids')
        records = [r for r in records if r['id'] in args.ids]
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    before = ROOT/'outputs/lote50_01/grid41'
    reference = {}
    for stage in ('peaks34', 'grid41'):
        for entry in records:
            folder = ROOT/'outputs/lote50_01'/stage/'images'/entry['id']
            for name in ('result.json', 'trace_masks.npz', 'signals.npz'):
                path = folder/name
                if path.is_file():
                    reference[str(path)] = {'sha256': digest(path), 'mtime_ns': path.stat().st_mtime_ns}
    manifest = {'entries': records, 'code': code_manifest(), 'references': reference,
                'before': 'grid41', 'control': 'peaks34', 'methods': {n:m.VERSION for n,m in METHODS.items()},
                'scope': 'Research only; original sources reprocessed, no copied extractions.'}
    write(out/'manifest.json', manifest)
    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_entry, (entry, str(out), str(before))): entry for entry in records}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = {'id': futures[future]['id'], 'error': str(error), 'traceback': traceback.format_exc()}
            done.append(result)
            write(out/'progress.json', {'done': len(done), 'total': len(records)})
            write(out/'report.json', sorted(done, key=lambda r:r['id']))
            print(json.dumps({'id': result['id'], 'done': len(done), 'total': len(records),
                              'errors': sum(m.get('status') == 'ERROR' for m in result.get('methods', [])),
                              'error': result.get('error')}), flush=True)
    unchanged = code_manifest() == manifest['code'] and all(
        digest(p) == info['sha256'] and Path(p).stat().st_mtime_ns == info['mtime_ns']
        for p,info in reference.items())
    errors = sum(bool(r.get('error')) for r in done) + sum(m.get('status') == 'ERROR' for r in done for m in r.get('methods', []))
    equivalence = [m['baseline_equivalence'] for r in done for m in r.get('methods', []) if 'baseline_equivalence' in m]
    exact = len(equivalence) == len(records) and all(all(v is not False for v in e.values()) for e in equivalence)
    write(out/'complete.json', {'processed': len(done), 'total': len(records), 'errors': errors,
                              'code_and_references_unchanged': unchanged, 'baseline_equivalent': exact,
                              'visual_review_complete': False, 'production_ready': False})
    if errors or not unchanged or not exact:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
