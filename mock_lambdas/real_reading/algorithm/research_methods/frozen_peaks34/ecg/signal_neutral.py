"""Explicit per-lead dark neutral ink extraction on a colored grid.

Geometry/calibration and other leads are preserved. No waveform template,
interpolation or source filename is used to recover missing pixels.
"""
import cv2
import numpy as np
from .signal import trace_lead


def ink_evidence(image):
    f=image.astype(np.float32)
    light=f.max(2)
    chroma=light-f.min(2)
    # Colored grid intersections survive the standard local-contrast filter.
    # Require measured dark, near-neutral ink instead of local contrast alone.
    raw=((light<180)&(chroma<50)).astype('uint8')
    return raw,255-light


def extract_lead(raw,contrast,lead):
    local=dict(lead)
    x0,x1=lead['trace_x'];step=lead['row_spacing_px']
    base=lead['baseline_y_px']
    lo=max(0,int(base-step*.25));hi=min(len(raw),int(base+step*.25)+1)
    profile=raw[lo:hi,x0:x1].sum(1)
    if profile.any():local['baseline_y_px']=int(lo+np.argmax(profile))
    trace=trace_lead(raw,contrast,local)
    x,y=trace['mask_origin'];h,w=trace['mask'].shape
    measured=raw[y:y+h,x:x+w]
    # A dense, thick vertical rectangle at a segment boundary is a separator,
    # not a QRS. Withhold its columns rather than inventing the hidden trace.
    vertical=cv2.morphologyEx(measured,cv2.MORPH_OPEN,
                            np.ones((max(9,int(step*.14)),1),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(vertical,8)
    excluded=np.zeros(w,bool)
    for k in range(1,n):
        xx,yy,ww,hh,area=stats[k]
        at_edge=xx<step*.08 or xx+ww>w-step*.08
        if at_edge and ww>=max(4,int(step*.014)) and area/(ww*hh)>.9:
            excluded[max(0,xx-1):min(w,xx+ww+1)]=True
    # Recover measured sides/extrema belonging to the same connected stroke.
    # Vertical runs can have two edges; a single run-per-column path loses one.
    connected=measured.copy();connected[:,excluded]=0
    n,components,_,_=cv2.connectedComponentsWithStats(connected,8)
    touched=np.unique(components[trace['mask'] & (connected>0)])
    trace['mask']=np.isin(components,touched[touched!=0])
    return summarize(trace,excluded)


def summarize(trace,suspect):
    mask=trace['mask'];x,y0=trace['mask_origin'];w=mask.shape[1]
    center=np.full(w,np.nan,np.float32);low=center.copy();high=center.copy()
    observed=mask.any(0);ambiguous=trace['ambiguous'].copy() | suspect | ~observed
    for col in np.flatnonzero(observed):
        rows=np.flatnonzero(mask[:,col])+y0
        center[col]=np.median(rows);low[col]=rows[0];high[col]=rows[-1]
        ambiguous[col] |= high[col]-low[col]>6 or np.any(np.diff(rows)>1)
    valid=observed & ~ambiguous
    trace.update(y_px=center,y_min_px=low,y_max_px=high,observed=observed,
                 ambiguous=ambiguous,valid=valid,coverage=float(observed.mean()),
                 unambiguous_fraction=float(valid.mean()),
                 baseline_y_px=float(np.nanmedian(center)) if observed.any() else trace['baseline_y_px'],
                 method='neutral_ink_per_lead_continuity')
    return trace


def refine(image,leads,traces,names):
    raw,contrast=ink_evidence(image)
    occupied=np.zeros(raw.shape,bool)
    for lead,trace in zip(leads,traces):
        if lead['name'] in names:continue
        x,y=trace['mask_origin'];h,w=trace['mask'].shape
        occupied[y:y+h,x:x+w] |= trace['mask']
    report={};candidates={};counts=np.zeros(raw.shape,np.uint16)
    for i,lead in enumerate(leads):
        if lead['name'] not in names:continue
        trace=extract_lead(raw,contrast,lead)
        x,y=trace['mask_origin'];h,w=trace['mask'].shape
        counts[y:y+h,x:x+w]+=trace['mask'];candidates[i]=trace
    for i,trace in candidates.items():
        lead=leads[i];x,y=trace['mask_origin'];h,w=trace['mask'].shape
        conflict=trace['mask'] & (occupied[y:y+h,x:x+w] | (counts[y:y+h,x:x+w]>1))
        trace['mask'] &= ~conflict
        summarize(trace,conflict.any(0))
        traces[i]=trace
        report[lead['name']]={'method':trace['method'],'conflicting_pixels_withheld':int(conflict.sum()),
                             'limitation':'Steep columns and withheld separators remain uncertain; no interpolation.'}
    return report
