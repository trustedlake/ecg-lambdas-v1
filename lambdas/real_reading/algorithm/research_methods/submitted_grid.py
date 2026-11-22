"""Experimental calibration fallback for faint dotted grids on full JPEG pages.

Use long projections, not a lowered confidence threshold on short tiles.
All three disjoint strips must independently support the same physical period.
Only calibration evidence is filtered; extractor input RGB remains untouched.
"""
from copy import deepcopy
import cv2
import numpy as np
from .frozen_peaks34.ecg.grid import estimate_grid, periodicity


def review_grid(rgb, config):
    original=estimate_grid(rgb,config)
    if original['valid']:return original
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
    evidence=np.clip(255-gray,0,70)
    evidence[gray<160]=0
    count=config['major_interval_minor_squares']
    expected=rgb.shape[1]/config['grid_width_mm_hint']*config['minor_square_mm']*count
    result=deepcopy(original)
    result['initial_estimate']=deepcopy(original)
    result['method']='experimental_long_strip_dotted_grid'
    result['strip_measurements']={}
    for axis,key in ((0,'x'),(1,'y')):
        observed=[periodicity(part.mean(axis),expected) for part in np.array_split(evidence,3,axis=axis)]
        result['strip_measurements'][key]=observed
        if any(p is None for p in observed):return original
        periods=np.array([p[0] for p in observed]);scores=np.array([p[1] for p in observed])
        if min(scores)<.35 or np.ptp(periods)/np.median(periods)>.02:return original
        spacing=float(np.median(periods)/count)
        previous=original[key].get('spacing_px')
        if previous and abs(spacing-previous)/spacing>.05:return original
        profile=evidence.mean(axis);major=spacing*count
        phases=np.arange(0,major,.25)
        phase=float(max(phases,key=lambda p:np.mean(np.interp(np.arange(p,len(profile),major),np.arange(len(profile)),profile))))
        result[key]={'spacing_px':spacing,'pixels_per_mm':spacing/config['minor_square_mm'],
                     'phase_px':phase,'periodicity_score':float(min(scores)),
                     'relative_dispersion':float(np.std(periods)/np.mean(periods)),
                     'tile_agreement':1.,'valid':True}
        result['controls'][key]=[phase,phase+major]
    ratio=result['x']['spacing_px']/result['y']['spacing_px']
    if not .65<ratio<1.55:return original
    result.update(valid=True,status='experimental_consistent_long_strips')
    return result
