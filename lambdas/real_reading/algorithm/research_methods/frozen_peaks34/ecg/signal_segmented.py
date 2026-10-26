"""Opt-in measured ink refinement with printed segment boundaries.

No row clipping or interpolation. Existing non-target owners are immutable;
ambiguous competing pixels are withheld. Crossings remain heuristic.
"""
import cv2,numpy as np
from scipy.ndimage import median_filter
from ecg.signal_affine import ink_evidence as affine_ink
from ecg.signal_printed import extract_lead
from ecg.signal_neutral import summarize

def ink_evidence(image):
    raw,c=affine_ink(image)
    red=image[...,0].astype('float32');horizontal=median_filter(red,size=(1,31))-red
    chroma=image.max(2).astype(float)-image.min(2).astype(float)
    extra=(horizontal>25)&(red<115)&(chroma<25)
    long=cv2.morphologyEx(extra.astype('uint8'),cv2.MORPH_CLOSE,np.ones((9,1),np.uint8))
    long=cv2.morphologyEx(long,cv2.MORPH_OPEN,np.ones((220,1),np.uint8))>0
    combined=(raw.astype(bool)&(c>12)&(chroma<35))|extra
    connected=cv2.morphologyEx(combined.astype('uint8'),cv2.MORPH_CLOSE,np.ones((9,1),np.uint8))
    long|=cv2.morphologyEx(connected,cv2.MORPH_OPEN,np.ones((220,1),np.uint8))>0
    long=cv2.dilate(long.astype('uint8'),np.ones((3,3),np.uint8))>0
    return (combined&~long).astype('uint8'),np.maximum(c,horizontal)

def refine(image,leads,traces,names):
    raw,c=ink_evidence(image);occupied=np.zeros(raw.shape,bool);candidates={};counts=np.zeros(raw.shape,int);report={}
    for i,(l,t) in enumerate(zip(leads,traces)):
        if l['name'] not in names:
            x,y=t['mask_origin'];h,w=t['mask'].shape;occupied[y:y+h,x:x+w]|=t['mask'];continue
        local=raw.copy();x0,x1=l['trace_x'];base=l['baseline_y_px'];step=l['row_spacing_px'];left=x0;right=x1
        # Narrow solid printed bars around the baseline delimit time segments.
        y0=max(0,int(base-step*.23));y1=min(len(raw),int(base+step*.23))
        dark=(image[y0:y1,x0:x1].max(2)<95).astype('uint8')
        vertical=cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((max(12,int(step*.13)),1),np.uint8))
        n,cc,stats,_=cv2.connectedComponentsWithStats(vertical,8)
        for k in range(1,n):
            xx,yy,ww,hh,area=stats[k]
            if ww<4 or hh>step*.38 or area/(ww*hh)<.85:continue
            if xx<(x1-x0)*.22:left=max(left,x0+xx+ww+2)
            elif xx+ww>(x1-x0)*.78:right=min(right,x0+xx-2)
        local[:,:left]=0;local[:,right:]=0
        if sum(v['row']==l['row'] for v in leads)>1:
            local[max(0,int(t['baseline_y_px']+step*.08)): ,left:min(right,int(left+step*.20))]=0
        candidate=extract_lead(local,c,l)
        x,y=candidate['mask_origin'];h,w=candidate['mask'].shape
        candidates[i]=candidate;counts[y:y+h,x:x+w]+=candidate['mask']
        report[l['name']]={'segment_x':[int(left),int(right)]}
    for i,t in candidates.items():
        x,y=t['mask_origin'];h,w=t['mask'].shape;t['mask'] &= ~occupied[y:y+h,x:x+w] & (counts[y:y+h,x:x+w]==1)
        old=np.zeros(raw.shape,bool);ox,oy=traces[i]['mask_origin'];oh,ow=traces[i]['mask'].shape;old[oy:oy+oh,ox:ox+ow]=traces[i]['mask']
        summarize(t,(old[y:y+h,x:x+w]!=t['mask']).any(0))
        t['method']='segmented_measured_ink_continuity'
        traces[i]=t
    return report
