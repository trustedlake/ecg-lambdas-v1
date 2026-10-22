"""Appearance-based trace segmentation with continuity; no waveform diagnosis."""
import cv2
import numpy as np
from .config import COLORS


def _page_edge_components(labels, stats):
    """Identify smooth, thin borders clipped by the top/bottom of the page.

    Interior horizontal traces are retained. A component with excursions or
    attached waveform strokes is retained too; this is not a flatline filter.
    All distances scale with the working image, independent of lead or file.
    """
    h,w=labels.shape
    excluded=[]
    for k in range(1,len(stats)):
        x,y,width,height,area=stats[k]
        if width < .5*w or height > .05*h or not (y==0 or y+height==h):
            continue
        component=labels[y:y+height,x:x+width]==k
        present=component.any(0)
        if present.mean()<.95:continue
        top=np.argmax(component,axis=0)
        bottom=height-1-np.argmax(component[::-1],axis=0)
        thickness=bottom-top+1
        if np.percentile(thickness[present],99)>max(4,.005*h):continue
        center=(top+bottom).astype(np.float32)/2
        # A local median follows a slowly bending paper edge, but not spikes.
        from scipy.ndimage import median_filter
        smooth=median_filter(center,size=max(5,int(.05*w)//2*2+1),mode='nearest')
        if np.percentile(abs(center[present]-smooth[present]),99)<=1:
            excluded.append(k)
    return excluded


def ink_evidence(a):
    # The red channel suppresses a red paper grid without recognizing a waveform.
    f=a.astype(np.float32);chroma=f.max(2)-f.min(2)
    gray=np.clip(f.max(2)+1.4*chroma,0,255)
    bg=cv2.GaussianBlur(gray,(0,0),7)
    contrast=np.maximum(0,bg-gray)
    norm=cv2.divide(gray,bg+1,scale=255)
    raw=((contrast>6)&(norm<241)).astype('uint8')
    if raw.mean()>.08:
        # Dense monochrome grids: subtract sustained row/column evidence.
        # This can leave gaps at intersections; never fill them with invented samples.
        horizontal=np.percentile(contrast,40,axis=1)[:,None]
        vertical=np.percentile(contrast,40,axis=0)[None,:]
        contrast=np.maximum(0,contrast-np.maximum(horizontal,vertical))
        raw=((contrast>6)&(norm<241)).astype('uint8')
    n,labels,stats,_=cv2.connectedComponentsWithStats(raw,8)
    keep=(stats[:,cv2.CC_STAT_AREA]>=4);keep[0]=False
    keep[_page_edge_components(labels,stats)]=False
    raw=keep[labels].astype('uint8')
    return raw,contrast


def locate_leads(raw,config,geometry):
    h,w=raw.shape
    priors=config['baseline_fractions']
    if geometry['method']!='colored_grid_boundary':
        priors=config.get('paper_baseline_fractions',priors)
    nrows=len(priors);expected=np.array(priors)*h
    profile=raw[:,int(.10*w):int(.96*w)].mean(1).astype('float32')
    profile=cv2.GaussianBlur(profile.reshape(-1,1),(1,5),0).ravel()
    step=float(np.median(np.diff(expected))) if nrows>1 else h*.5
    centers=[]
    for p in expected:
        lo=max(0,int(p-step*.30));hi=min(h,int(p+step*.30))
        ys=np.arange(lo,hi)
        score=profile[lo:hi]-.035*((ys-p)/(step*.30))**2
        centers.append(int(lo+np.argmax(score)))
    if any(b<=a for a,b in zip(centers,centers[1:])):raise ValueError('Invalid baseline ordering')
    lo_x,hi_x=config.get('signal_x_fraction',[.04,.97]);left=int(lo_x*w);right=int(hi_x*w)
    cols=config['columns'];leads=[]
    matrix=config['leads']+[[n] for n in config.get('rhythm_leads',[])]
    for row,names in enumerate(matrix):
        top=max(0,int((centers[row-1]+centers[row])/2)) if row else max(0,int(centers[row]-step*.55))
        bottom=min(h,int((centers[row]+centers[row+1])/2)) if row+1<nrows else min(h,int(centers[row]+step*.50))
        for col,name in enumerate(names):
            x0=int(left+(right-left)*col/len(names));x1=int(left+(right-left)*(col+1)/len(names))
            margin=max(3,int(w*config.get('label_exclusion_fraction',.008)))
            # Per-lead baseline refines the row estimate while preserving ordering.
            local=raw[max(0,centers[row]-int(step*.18)):min(h,centers[row]+int(step*.18)+1),x0+margin:x1].sum(1)
            baseline=max(0,centers[row]-int(step*.18))+int(np.argmax(local))
            leads.append({'name':name,'row':row,'column':col,'color':COLORS[name],
                          'box':[x0,top,x1,bottom],'trace_x':[x0+margin,x1-2],
                          'baseline_y_px':baseline,'row_spacing_px':step})
    return leads


def trace_lead(raw,contrast,lead):
    h,w=raw.shape;x0,x1=lead['trace_x'];base=lead['baseline_y_px'];step=lead['row_spacing_px']
    # Overlap beyond presentation boxes is intentional, to retain high QRS peaks.
    y0=max(0,int(base-step*.80));y1=min(h,int(base+step*.65))
    roi=raw[y0:y1,x0:x1].copy();height,width=roi.shape
    n,cc,stats,_=cv2.connectedComponentsWithStats(roi,8)
    keep=np.zeros(n,bool)
    for k in range(1,n):
        xx,yy,ww,hh,area=stats[k]
        near=(yy<=base-y0+step*.16 and yy+hh>=base-y0-step*.16)
        keep[k]=near and ((ww>=7 and area>=10) or (hh>=step*.18 and area>=12))
    mask=keep[cc]
    # Candidate runs preserve their envelope for downstream uncertainty review.
    kmax=8;ys=np.full((width,kmax),np.nan,np.float32);cost=np.full((width,kmax),1e6,np.float32)
    lower=np.full_like(ys,np.nan);upper=np.full_like(ys,np.nan);supported=np.zeros_like(ys,bool)
    ambiguity=np.zeros(width,bool)
    for x in range(width):
        points=np.flatnonzero(mask[:,x])
        if points.size:
            runs=np.split(points,np.flatnonzero(np.diff(points)>1)+1)
            runs=sorted(runs,key=lambda r:float(contrast[r+y0,x+x0].mean())-.10*abs(float(np.median(r))+y0-base),reverse=True)[:kmax]
            ambiguity[x]=len(runs)>1 and np.ptp([np.median(r) for r in runs])>step*.18
            for j,r in enumerate(runs):
                mid=float(np.median(r))+y0;ys[x,j]=mid;lower[x,j]=r[0]+y0;upper[x,j]=r[-1]+y0
                strength=float(contrast[r+y0,x+x0].mean());cost[x,j]=-min(60,strength)*.11+abs(mid-base)/step*.65
                supported[x,j]=True
        else:
            ys[x,0]=base;cost[x,0]=3 # Internal path only; exported value remains NaN.
    back=np.zeros((width,kmax),np.int8);previous=cost[0].copy()
    for x in range(1,width):
        distance=abs(ys[x,:,None]-ys[x-1,None,:]);distance=np.nan_to_num(distance,nan=1e5)
        transition=np.minimum(distance,step*.8)*.065
        total=transition+previous[None,:]
        back[x]=np.argmin(total,axis=1);previous=cost[x]+np.min(total,axis=1)
    choice=np.zeros(width,np.int8);choice[-1]=int(np.argmin(previous))
    for x in range(width-1,0,-1):choice[x-1]=back[x,choice[x]]
    idx=np.arange(width);observed=supported[idx,choice]
    y=ys[idx,choice];y[~observed]=np.nan
    low=lower[idx,choice];high=upper[idx,choice]
    # Median is only a geometric reference, not a clinically established isoelectric line.
    baseline=float(np.nanmedian(y)) if observed.any() else float(base)
    selected=np.zeros_like(mask)
    for x in np.flatnonzero(observed):
        selected[int(low[x])-y0:int(high[x])-y0+1,x]=mask[int(low[x])-y0:int(high[x])-y0+1,x]
    valid=observed & ~ambiguity
    return {'x_px':np.arange(x0,x1,dtype=np.float32),'y_px':y,'y_min_px':low,'y_max_px':high,
            'observed':observed,'valid':valid,'ambiguous':ambiguity,'baseline_y_px':baseline,
            'mask':selected,'mask_origin':[x0,y0],
            'coverage':float(np.mean(observed)),'unambiguous_fraction':float(np.mean(valid)),
            'method':'contrast_components_run_center_continuity',
            'baseline_method':'median_observed_trace_not_clinical_isoelectric_reference'}
