"""Non-destructive correction of extracted ECG pixels; never clinical validation."""
from pathlib import Path
import hashlib, json
import cv2
import numpy as np
from PIL import Image
from .pipeline import write_json


def correct(previous_dir, output_dir, lead_name, positive_png=None, negative_png=None):
    """Masks are grayscale PNGs, white=selected, in *working-image* coordinates.

    Positive is intersected with locally detected image ink (never drawn as an
    invented trace). Negative removes existing mask. Both are restricted to the
    chosen lead mask ROI; changed columns are invalidated for clinical use.
    """
    previous_dir, output_dir = Path(previous_dir), Path(output_dir)
    if previous_dir.resolve() == output_dir.resolve():
        raise ValueError('Correction must write to a NEW directory')
    if output_dir.exists():
        raise FileExistsError('Correction output must not already exist')
    meta = json.loads((previous_dir/'result.json').read_text(encoding='utf8'))
    if meta['status'].startswith('REJECTED_'):
        raise ValueError('Color-gate rejection cannot be overridden by manual painting. Request a new image.')
    record = next((r for r in meta['leads'] if r['name'] == lead_name), None)
    if record is None: raise ValueError('Unknown lead')
    if 'mask_origin' not in record or not (previous_dir/'working_image.png').is_file():
        raise ValueError('Original run predates editing support; re-run analysis using this package')
    original_img = np.asarray(Image.open(previous_dir/'working_image.png').convert('RGB'))
    h, w = original_img.shape[:2]
    def load_mask(path):
        if path is None: return np.zeros((h,w), bool)
        m = np.asarray(Image.open(path).convert('L'))
        if m.shape != (h,w): raise ValueError('Paint mask dimensions do not match working image')
        return m >= 128
    positive, negative = load_mask(positive_png), load_mask(negative_png)
    # A manually modified waveform is never an automatically valid clinical sample.
    if not (positive|negative).any(): raise ValueError('Empty correction')
    if (positive & negative).any(): raise ValueError('Positive and negative paint cannot overlap')
    with np.load(previous_dir/'trace_masks.npz') as masks:
        old_masks={k:masks[k].astype(bool) for k in masks.files}
    with np.load(previous_dir/'signals.npz') as signals:
        arrays={k:signals[k].copy() for k in signals.files}
    old = old_masks[lead_name]
    x0,y0 = map(int,record['mask_origin']); mh,mw=old.shape
    if x0<0 or y0<0 or x0+mw>w or y0+mh>h: raise ValueError('Invalid mask origin')
    region=np.s_[y0:y0+mh,x0:x0+mw]
    if (positive|negative).sum() != (positive[region]|negative[region]).sum():
        raise ValueError('Painting outside selected lead ROI is not allowed')
    # Restrict additions to measured ink, not a hand-drawn replacement waveform.
    from .signal import ink_evidence
    evidence,_=ink_evidence(original_img)
    additions=positive[region] & (evidence[region]>0)
    # Do not steal a pixel owned by a different lead.
    occupied=np.zeros((h,w),bool)
    for k,other in old_masks.items():
        if k==lead_name: continue
        other_record=next(r for r in meta['leads'] if r['name']==k)
        ox,oy=map(int,other_record['mask_origin']); oh,ow=other.shape
        occupied[oy:oy+oh,ox:ox+ow] |= other
    additions &= ~occupied[region]
    updated=(old | additions) & ~negative[region]
    changed=old^updated
    if not changed.any(): raise ValueError('Correction changed no measurable trace pixels')
    prefix=lead_name+'__'; x=arrays[prefix+'x_px']; center=arrays[prefix+'y_px']
    if len(x)!=mw: raise ValueError('Mismatched exported signal width')
    changed_columns=changed.any(axis=0)
    for col in np.flatnonzero(changed_columns):
        ys=np.flatnonzero(updated[:,col])+y0
        if ys.size:
            arrays[prefix+'y_px'][col]=float(np.median(ys))
            arrays[prefix+'y_min_px'][col]=float(ys.min())
            arrays[prefix+'y_max_px'][col]=float(ys.max())
        else:
            for key in ('y_px','y_min_px','y_max_px'):
                arrays[prefix+key][col]=np.nan
        arrays[prefix+'observed'][col]=bool(ys.size)
        arrays[prefix+'ambiguous'][col]=True
        arrays[prefix+'selected_valid'][col]=False
        arrays[prefix+'valid'][col]=False
        arrays[prefix+'analysis_mv'][col]=np.nan
        for key in ('amplitude_mv','amplitude_min_mv','amplitude_max_mv'):
            arrays[prefix+key][col]=np.nan
    old_masks[lead_name]=updated
    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir/'signals.npz',**arrays)
    np.savez_compressed(output_dir/'trace_masks.npz',**old_masks)
    Image.fromarray(original_img).save(output_dir/'working_image.png')
    # Overlay diff on original rectified image: green retained, red added, blue removed.
    overlay=original_img.copy()
    roi=overlay[region]
    roi[updated&old]=(15,185,65)
    roi[updated&~old]=(235,45,45)
    roi[old&~updated]=(40,100,230)
    Image.fromarray(overlay).save(output_dir/'correction_preview.png')
    import shutil
    shutil.copyfile(previous_dir/'visao_geral.png', output_dir/'visao_geral_original.png')
    record['coverage']=float(updated.any(axis=0).mean())
    record['valid_samples']=int(arrays[prefix+'valid'].sum())
    meta['status']='MANUAL_REVIEW_REQUIRED'
    meta['warnings']=sorted(set(meta.get('warnings',[])+['MANUAL_CORRECTION_UNVERIFIED']))
    meta['manual_correction']={'lead':lead_name,'parent_result_sha256':hashlib.sha256((previous_dir/'result.json').read_bytes()).hexdigest(),
        'added_pixels':int((updated&~old).sum()),'removed_pixels':int((old&~updated).sum()),
        'changed_columns':int(changed_columns.sum()),'working_image_coordinates':True,
        'limitations':'Changed columns marked ambiguous and invalid; no automated clinical approval.'}
    meta['source_result_directory']=str(previous_dir)
    write_json(output_dir/'result.json',meta)
    return meta
