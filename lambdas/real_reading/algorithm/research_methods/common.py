"""Shared measured tracing for candidates, with explicit uncertainty."""
import numpy as np
import cv2
from .contracts import Extraction
from .frozen_peaks34.ecg import signal_tinted, trace_ownership


def trace_evidence(context, raw, contrast):
    raw = np.asarray(raw, dtype=np.uint8)
    if context.geometry.get('extraction_domain') == 'full_photo':
        return trace_full_photo(context, raw, contrast)
    leads = signal_tinted.locate_leads(raw, context.config, context.geometry)
    traces = [signal_tinted.trace_lead(raw, contrast, lead) for lead in leads]
    ownership = trace_ownership.separate_traces(traces, leads, context.image.shape)
    return Extraction(leads, traces, {'ownership': ownership})


def locate_full_photo(raw, context):
    """Locate anchors in the layout frame; extraction still sees the entire photo."""
    x0,y0,x1,y1=context.geometry.get('layout_box',context.geometry['calibration_box'])
    leads=signal_tinted.locate_leads(raw[y0:y1,x0:x1],context.config,context.geometry)
    for lead in leads:
        lead['box']=[v+(x0 if i%2==0 else y0) for i,v in enumerate(lead['box'])]
        lead['trace_x']=[v+x0 for v in lead['trace_x']]
        lead['baseline_y_px']+=y0
    return leads


def trace_full_photo(context, raw, contrast, leads=None):
    """Assign connected measured strokes to sustained baseline anchors.

    Presentation rows never bound a component. A component connected to multiple
    lead anchors is withheld rather than guessed by distance to the nearest row.
    Closing is connectivity-only: every exported pixel exists in raw evidence.
    """
    raw=np.asarray(raw,dtype=np.uint8)
    leads=locate_full_photo(raw,context) if leads is None else leads
    h,w=raw.shape
    graph=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(graph,8)
    owners=np.full(n,-1,np.int16);claims=np.zeros(n,np.uint16)
    row_claims={}
    for row in sorted({lead['row'] for lead in leads}):
        row_ids=[]
        for lead in leads:
            if lead['row']!=row:continue
            x0,x1=lead['trace_x'];base=lead['baseline_y_px'];step=lead['row_spacing_px']
            y0=max(0,int(base-.12*step));y1=min(h,int(base+.12*step)+1)
            anchor=cv2.morphologyEx(raw[y0:y1,x0:x1],cv2.MORPH_OPEN,
                                    np.ones((1,max(7,int(step*.04))),np.uint8))>0
            row_ids.extend(np.unique(labels[y0:y1,x0:x1][anchor]).tolist())
        ids=np.array(sorted(set(row_ids)-{0}),dtype=int)
        claimed=np.zeros(n,bool);claimed[ids]=True;row_claims[row]=claimed
        owners[ids]=row;claims[ids]+=1
    owners[claims!=1]=-1
    traces=[]
    for index,lead in enumerate(leads):
        x0,x1=lead['trace_x']
        ids=labels[:,x0:x1]
        mask=(owners[ids]==lead['row'])&(raw[:,x0:x1]>0)
        disputed=(claims[ids]>1)&row_claims[lead['row']][ids]&(raw[:,x0:x1]>0)
        observed=mask.any(0);ambiguous=disputed.any(0)
        yy=np.full(x1-x0,np.nan,np.float32);low=yy.copy();high=yy.copy()
        for x in np.flatnonzero(observed):
            points=np.flatnonzero(mask[:,x])
            yy[x]=np.median(points);low[x]=points[0];high[x]=points[-1]
            # Distinct vertical runs may be two traces, not a single sample.
            if np.any(np.diff(points)>3):ambiguous[x]=True
        # Ambiguous columns must not export a fabricated numerical sample.
        yy[ambiguous]=np.nan
        traces.append(dict(x_px=np.arange(x0,x1,dtype=np.float32),y_px=yy,
            y_min_px=low,y_max_px=high,observed=observed,ambiguous=ambiguous,
            valid=observed&~ambiguous,baseline_y_px=float(lead['baseline_y_px']),
            mask=mask,mask_origin=[x0,0],coverage=float(observed.mean()),
            unambiguous_fraction=float((observed&~ambiguous).mean()),
            method='full_photo_unique_component_anchors',
            baseline_method='measured_layout_anchor'))
    return Extraction(leads,traces,{'ownership':dict(method='unique_component_anchors',
        ambiguous_components=int((claims>1).sum()),unassigned_components=int((claims==0).sum()),
        limitation='Crossings connected to multiple anchors are withheld; disconnected tips may be lost.')})
