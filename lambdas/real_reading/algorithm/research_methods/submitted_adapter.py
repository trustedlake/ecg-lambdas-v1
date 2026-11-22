"""Opt-in adapters for eight user-supplied scripts. Vendor sources stay unchanged.

Original native artifacts remain available. Scalar signals are column medians;
ordered paths and masks are retained, never represented as calibrated diagnoses.
No source identity enters an extractor, layout or threshold decision.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import cv2
import numpy as np
from .contracts import Extraction, MethodAbstention

VENDOR = Path(__file__).with_name('submitted_20261105')
NAMES = ('components_close', 'components_boxes', 'components_extents',
         'components_dividers', 'skeleton_path', 'components_bands', 'ink_path', 'band_mask')


def column_trace(points, width, height, name, full_x=None):
    """Keep absent columns NaN. Never interpolate a path across missing ink."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(1)]
    if not len(pts):
        raise MethodAbstention('No measured points for ' + name)
    xi = np.rint(pts[:, 0]).astype(int)
    yi = np.rint(pts[:, 1]).astype(int)
    if np.any((xi < 0) | (xi >= width) | (yi < 0) | (yi >= height)):
        raise ValueError('Native points outside image')
    lo, hi = full_x if full_x is not None else (int(xi.min()), int(xi.max()) + 1)
    x = np.arange(lo, hi, dtype=np.float32)
    y = np.full(len(x), np.nan, np.float32)
    ymin = y.copy(); ymax = y.copy()
    top, bottom = int(yi.min()), int(yi.max()) + 1
    mask = np.zeros((bottom-top, hi-lo), bool)
    for col in np.unique(xi):
        vals = pts[xi == col, 1]
        j = col-lo
        y[j] = np.median(vals); ymin[j] = vals.min(); ymax[j] = vals.max()
        mask[yi[xi == col]-top, j] = True
    observed = np.isfinite(y)
    ambiguous = observed & (ymax-ymin > 2)
    baseline = float(np.median(y[observed]))
    return dict(x_px=x, y_px=y, y_min_px=ymin, y_max_px=ymax,
                observed=observed, ambiguous=ambiguous, valid=observed & ~ambiguous,
                mask=mask, mask_origin=[lo, top], baseline_y_px=baseline,
                coverage=float(observed.mean()), unambiguous_fraction=float((observed & ~ambiguous).mean()))


class SubmittedMethod:
    def __init__(self, name):
        self.name = name
        self.path = VENDOR / (name + '.py')
        self.VERSION = 'submitted-' + hashlib.sha256(self.path.read_bytes()).hexdigest()[:12] + '-adapter1'

    def extract(self, context, artifact_dir=None):
        if context.config['leads'] != [['I','aVR','V1','V4'], ['II','aVL','V2','V5'], ['III','aVF','V3','V6']]:
            raise MethodAbstention('Submitted scripts require explicit 3x4 layout')
        with tempfile.TemporaryDirectory(prefix='ecg_submitted_') as tmp:
            target = Path(artifact_dir) if artifact_dir else Path(tmp)/'artifacts'
            target.mkdir(parents=True, exist_ok=True)
            source = Path(tmp)/'input.png'
            bgr = cv2.cvtColor(context.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(source), bgr)
            args = {
                'components_close': ['--output', str(target)],
                'components_boxes': ['-o', str(target)],
                'components_extents': ['--overlay',str(target/'overlay.png'),'--mask',str(target/'mask.png'),'--csv',str(target/'points.csv')],
                'components_dividers': ['--output',str(target/'overlay.png')],
                'skeleton_path': ['--output',str(target/'overlay.png'),'--csv',str(target/'points.csv')],
                'components_bands': ['--output-dir',str(target)],
                'ink_path': ['--output',str(target/'overlay.png')],
                'band_mask': ['--saida',str(target/'overlay.png'),'--mascara',str(target/'mask.png')],
            }[self.name]
            process = subprocess.run([sys.executable,'-B',str(self.path),str(source),*args],
                                     capture_output=True, text=True, encoding='utf8', errors='replace', timeout=120)
            (target/'native.log').write_text(process.stdout+'\n'+process.stderr, encoding='utf8')
            if process.returncode:
                raise MethodAbstention('Native extraction failed: '+process.stderr[-1000:])
            points = {}; domains = {}; synthetic = 0
            def add(name, x, y):
                if name not in context.config['leads'][0]+context.config['leads'][1]+context.config['leads'][2]:
                    name = 'rhythm_unassigned'  # do not relabel II/III from hardcoded script text
                points.setdefault(name, []).append((float(x),float(y)))
            if self.name == 'components_close':
                with np.load(target/'ecg_derivacoes.npz') as masks:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    for name in masks.files:
                        raw = masks[name] > 0
                        synthetic += int((raw & (gray >= 125)).sum())
                        yy, xx = np.where(raw & (gray < 125))
                        for x,y in zip(xx,yy): add(name,x,y)
            elif self.name == 'ink_path':
                data=json.loads((target/'overlay_points.json').read_text(encoding='utf8'))
                for lead in data['leads']:
                    for x,y in lead['points']: add(lead['lead'],x,y)
            elif self.name == 'band_mask':
                spec=importlib.util.spec_from_file_location('_submitted_band',self.path)
                mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
                raw,bases=mod.extrair_mascara_ecg(bgr)
                h,w=raw.shape
                cuts=[0]+[(a+b)//2 for a,b in zip(bases,bases[1:])]+[h]
                xa,xb=int(w*.033),int(w*.98)
                for r in range(4):
                    cols=4 if r<3 else 1
                    for c in range(cols):
                        x0=round(xa+(xb-xa)*c/cols);x1=round(xa+(xb-xa)*(c+1)/cols)
                        yy,xx=np.where(raw[cuts[r]:cuts[r+1],x0:x1]>0)
                        name=context.config['leads'][r][c] if r<3 else 'rhythm_unassigned'
                        domains[name]=(x0,x1)
                        for x,y in zip(xx+x0,yy+cuts[r]):add(name,x,y)
            else:
                csv_name={'components_boxes':'pontos_ecg.csv','components_extents':'ecg_pixels.csv',
                          'components_dividers':'overlay_pontos.csv','skeleton_path':'points.csv',
                          'components_bands':'ecg_pontos.csv'}[self.name]
                if self.name=='components_extents':csv_name='points.csv'
                records=list(csv.DictReader((target/csv_name).open(encoding='utf8')))
                for row in records:
                    name=row.get('lead',row.get('derivacao'))
                    x=float(row['x_px'])
                    key=name if name in sum(context.config['leads'],[]) else 'rhythm_unassigned'
                    a,b=domains.get(key,(int(x),int(x)+1));domains[key]=(min(a,int(x)),max(b,int(x)+1))
                    y=row.get('y_px',row.get('y_median_px',row.get('y_central_px')))
                    if y not in ('',None):add(name,x,y)
                if self.name=='components_boxes':
                    # Native closing pixels stay in original artifacts, not in measured evidence.
                    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                    for name, values in points.items():
                        kept=[]
                        for x,y in values:
                            if gray[round(y),round(x)]<85: kept.append((x,y))
                            else: synthetic+=1
                        points[name]=kept
            h,w=context.image.shape[:2]
            leads=[];traces=[]
            for name,values in points.items():
                if not values:continue
                trace=column_trace(values,w,h,name,domains.get(name))
                leads.append({'name':name,'baseline_y_px':trace['baseline_y_px']})
                traces.append(trace)
            if not traces:raise MethodAbstention('No measured traces')
            return Extraction(leads,traces,{'native_source_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'adapter':'column median; no gap interpolation; native paths/masks retained',
                'removed_unmeasured_samples_or_pixels':synthetic,
                'rhythm':'unassigned and excluded from standard 12-lead selector',
                'activation':'explicit 3x4 research batch; no validated appearance gate'})


SUBMITTED_METHODS={'submitted_'+name:SubmittedMethod(name) for name in NAMES}
