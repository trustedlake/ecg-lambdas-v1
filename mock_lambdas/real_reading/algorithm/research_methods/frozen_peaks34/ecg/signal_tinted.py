"""Opt-in recovery for faint colored traces; validate each new appearance family.

The standard path remains the implementation behind the accepted gallery.
This profile reduces chroma suppression and jointly locates rows, with a local
sloping baseline and bounded darkness costs to avoid selecting the paper edge.
No reference waveform, filename, or prior result is used during extraction.
"""
import cv2
import numpy as np
from .config import COLORS


def stroke_evidence(a):
    """Measured narrow dark strokes after cancelling local grid background."""
    red=a[...,0].astype(np.float32)
    narrow=np.maximum(0,cv2.blur(red,(31,1))-red)
    detail=np.maximum(0,narrow-1.5*cv2.blur(narrow,(1,151)))
    return (detail>7)&(narrow>10)&(red<242),detail


def _chromatic_channel(gray):
    """Local measured contrast, with sustained dense-grid evidence discounted."""
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
    return raw,contrast


def ink_evidence(a):
    # Keep the permissive evidence used for weak purple baselines. Reducing that
    # mask to the stricter scoring channel would erase faint measured samples.
    f=a.astype(np.float32);chroma=f.max(2)-f.min(2)
    gray=np.clip(f.max(2)+0.7*chroma,0,255)
    raw,_=_chromatic_channel(gray)
    # Red grid pixels can be darker than a purple waveform. Penalise red excess
    # when scoring alternative paths, while retaining the original evidence.
    # The narrow-stroke channel below restores contrast at grid crossings.
    red_excess=np.maximum(0,(f[...,0]+f[...,1])*.5-f[...,2])
    score_gray=np.clip(f.max(2)+0.7*chroma+2*red_excess,0,255)
    _,contrast=_chromatic_channel(score_gray)
    # A red grid crossing increases chroma even on a dark waveform pixel, making
    # the chroma channel clip to white. Recover narrow strokes from local red
    # contrast: horizontal grid cancels against its own row background; sustained
    # vertical evidence is penalised. This supplies measured pixels, not bridges.
    extra,detail=stroke_evidence(a)
    raw=np.maximum(raw,extra.astype('uint8'))
    contrast=np.maximum(contrast,np.where(extra,detail,0))
    n,labels,stats,_=cv2.connectedComponentsWithStats(raw,8)
    keep=(stats[:,cv2.CC_STAT_AREA]>=4);keep[0]=False
    raw=keep[labels].astype('uint8')
    return raw,contrast


def recover_stroke_pixels(extra,trace,lead):
    """Retain both sides of a steep stroke, not just one run per column.

    Only narrow, measured components touching the selected trace qualify.
    Long grid lines and broad horizontal branches cannot be recovered this way.
    """
    x0,y0=trace['mask_origin'];selected=trace['mask'].copy();h,w=selected.shape
    measured=extra[y0:y0+h,x0:x0+w].astype('uint8')
    # Separate a vertical stroke from the horizontal waveform it joins. Otherwise
    # an entire row forms one broad component and its upstroke is rejected.
    support=cv2.morphologyEx(measured,cv2.MORPH_OPEN,np.ones((7,1),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(support,8)
    touching=cv2.dilate(selected.astype('uint8'),np.ones((3,3),np.uint8))>0
    ids=np.unique(labels[touching]);keep=np.zeros(n,bool);step=lead['row_spacing_px']
    for k in ids:
        if k==0:continue
        xx,yy,ww,hh,area=stats[k]
        keep[k]=(hh>=max(7,1.2*ww) and hh<=1.5*step and ww<=.45*step)
    recovered=keep[labels].astype('uint8')
    recovered=(cv2.dilate(recovered,np.ones((3,3),np.uint8))>0)&(measured>0)
    selected|=recovered
    observed=selected.any(0)
    y=np.full(w,np.nan,np.float32);lo=y.copy();hi=y.copy()
    for x in np.flatnonzero(observed):
        points=np.flatnonzero(selected[:,x])+y0
        y[x]=np.median(points);lo[x]=points[0];hi[x]=points[-1]
    valid=observed & ~trace['ambiguous']
    trace.update(mask=selected,y_px=y,y_min_px=lo,y_max_px=hi,observed=observed,valid=valid,
                 baseline_y_px=float(np.nanmedian(y)) if observed.any() else trace['baseline_y_px'],
                 coverage=float(observed.mean()),unambiguous_fraction=float(valid.mean()))
    return trace


def locate_leads(raw,config,geometry):
    h,w=raw.shape
    priors=config['baseline_fractions']
    if geometry['method']!='colored_grid_boundary':
        priors=config.get('paper_baseline_fractions',priors)
    nrows=len(priors);expected=np.array(priors)*h
    profile=raw[:,int(.10*w):int(.96*w)].mean(1).astype('float32')
    profile=cv2.GaussianBlur(profile.reshape(-1,1),(1,5),0).ravel()
    step=float(np.median(np.diff(expected))) if nrows>1 else h*.5
    if config['columns'] == 1 and nrows >= 6:
        # Fit all row positions together. Independent searches can skip a
        # weak lead and select the footer or the paper edge when the printed row
        # spacing varies.
        best_score = -np.inf
        best_expected = expected
        for scale in np.linspace(.82, 1.08, 27):
            for offset in np.linspace(-.35, .35, 29):
                candidate = expected[0] + (expected - expected[0]) * scale + offset * step
                if candidate[0] < 0 or candidate[-1] >= h:
                    continue
                strengths = []
                for center in candidate:
                    lo = max(0, int(center - step * .10))
                    hi = min(h, int(center + step * .10) + 1)
                    strengths.append(float(profile[lo:hi].max()))
                score = np.mean(strengths) + .5 * min(strengths) - .025 * abs(scale - 1) - .015 * abs(offset)
                if score > best_score:
                    best_score, best_expected = score, candidate
        expected = best_expected
        step = float(np.median(np.diff(expected)))
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
    # A slightly rotated photo tilts the whole row. Fit a straight local baseline
    # through the densest ink row of each chunk, so pieces far from a horizontal
    # reference are not discarded. Falls back to horizontal when the fit is weak.
    bx=np.full(x1-x0,float(base))
    lo_b=max(0,int(base-step*.20));hi_b=min(h,int(base+step*.20)+1);xc=[];yc=[];wc=[]
    for c0 in range(x0,x1,max(8,(x1-x0)//12)):
        rows=cv2.GaussianBlur(raw[lo_b:hi_b,c0:min(x1,c0+max(8,(x1-x0)//12))].sum(1).astype('float32').reshape(-1,1),(1,5),0).ravel()
        if rows.max()>0:xc.append(c0-x0+(x1-x0)/24);yc.append(lo_b+int(np.argmax(rows)));wc.append(float(rows.max()))
    if len(xc)>=4:
        slope,icpt=np.polyfit(xc,yc,1,w=np.sqrt(wc))
        line=icpt+slope*np.arange(x1-x0)
        if abs(slope)*(x1-x0)<step*.6 and np.median(np.abs(np.array(yc)-(icpt+slope*np.array(xc))))<step*.08:
            bx=line-np.median(line)+base
    # Overlap beyond presentation boxes is intentional, to retain high QRS peaks
    # that cross into the neighbouring row.
    y0=max(0,int(base-step*1.25));y1=min(h,int(base+step*1.10))
    roi=raw[y0:y1,x0:x1].copy();height,width=roi.shape
    # Grid lines cut steep QRS strokes into short pieces. A vertical closing is used
    # only to decide which pieces belong to a component near the baseline; the
    # exported mask keeps original pixels only.
    # A vertical grid line spans several rows of the page; a QRS stroke does not.
    # Columns lying on such a line are never bridged.
    gridlike=np.zeros(width,bool)
    for x in range(width):
        points=np.flatnonzero(raw[:,x+x0])
        if points.size:
            pieces=np.split(points,np.flatnonzero(np.diff(points)>7)+1)
            gridlike[x]=max(p[-1]-p[0] for p in pieces)>step*3
    bridged=cv2.morphologyEx(roi,cv2.MORPH_CLOSE,np.ones((7,1),np.uint8))
    bridged[:,gridlike]=roi[:,gridlike]
    n,cc,stats,_=cv2.connectedComponentsWithStats(bridged,8)
    keep=np.zeros(n,bool)
    for k in range(1,n):
        xx,yy,ww,hh,area=stats[k]
        b=bx[min(len(bx)-1,xx+ww//2)]
        near=(yy<=b-y0+step*.16 and yy+hh>=b-y0-step*.16)
        keep[k]=near and ((ww>=7 and area>=10) or (hh>=step*.18 and area>=12))
    mask=keep[cc]&(roi>0)
    # Typical trace contrast: the strongest ink at the baseline band of each column.
    band=contrast[max(0,base-int(step*.16)):base+int(step*.16)+1,x0:x1]
    trace_contrast=float(np.median(band.max(0))) if band.size else 0.
    # Ambiguity keeps its earlier definition (own band, unbridged components, unjoined
    # runs), so the wider window and joined strokes only change the selection.
    a0=max(0,int(base-step*.80));a1=min(h,int(base+step*.65))
    n2,cc2,stats2,_=cv2.connectedComponentsWithStats(raw[a0:a1,x0:x1],8)
    keep2=np.zeros(n2,bool)
    for k in range(1,n2):
        xx,yy,ww,hh,area=stats2[k]
        b=bx[min(len(bx)-1,xx+ww//2)]
        near=(yy<=b-a0+step*.16 and yy+hh>=b-a0-step*.16)
        keep2[k]=near and ((ww>=7 and area>=10) or (hh>=step*.18 and area>=12))
    own=keep2[cc2]
    ambiguity=np.zeros(width,bool)
    for x in range(width):
        points=np.flatnonzero(own[:,x])
        if points.size:
            runs=np.split(points,np.flatnonzero(np.diff(points)>1)+1)
            runs=sorted(runs,key=lambda r:float(contrast[r+a0,x+x0].mean())-.10*abs(float(np.median(r))+a0-bx[x]),reverse=True)[:8]
            ambiguity[x]=len(runs)>1 and np.ptp([np.median(r) for r in runs])>step*.18
    # Candidate runs preserve their envelope for downstream uncertainty review.
    kmax=8;ys=np.full((width,kmax),np.nan,np.float32);cost=np.full((width,kmax),1e6,np.float32)
    lower=np.full_like(ys,np.nan);upper=np.full_like(ys,np.nan);supported=np.zeros_like(ys,bool)
    for x in range(width):
        points=np.flatnonzero(mask[:,x])
        if points.size:
            # Pieces of one stroke separated by a grid-line gap stay one run, but only
            # when the joined run is as dark as the trace; faint grid lines stay split.
            runs=[]
            for r in np.split(points,np.flatnonzero(np.diff(points)>(1 if gridlike[x] else 7))+1):
                if r.size>1 and float(contrast[r+y0,x+x0].mean())<.65*trace_contrast:
                    runs.extend(np.split(r,np.flatnonzero(np.diff(r)>1)+1))
                else:runs.append(r)
            runs=sorted(runs,key=lambda r:float(contrast[r+y0,x+x0].mean())-.10*abs(float(np.median(r))+y0-bx[x]),reverse=True)[:kmax]
            for j,r in enumerate(runs):
                mid=float(np.median(r))+y0;ys[x,j]=mid;lower[x,j]=r[0]+y0;upper[x,j]=r[-1]+y0
                strength=float(contrast[r+y0,x+x0].mean())
                # Darkness counts up to a little above this lead's own trace, so a
                # much darker straight line (paper edge, border) cannot win on it alone.
                cap=min(60.,1.2*trace_contrast) if trace_contrast>0 else 60.
                cost[x,j]=-min(cap,strength)*.11+abs(mid-bx[x])/step*.65
                # A run that starts at the baseline (a QRS stroke) is not far; a run
                # lying entirely near the next row (its baseline) is strongly penalised.
                gap=max(0.,lower[x,j]-bx[x],bx[x]-upper[x,j]);cost[x,j]+=15*max(0.,gap/step-.30)
                supported[x,j]=True
        else:
            ys[x,0]=bx[x];cost[x,0]=3 # Internal path only; exported value remains NaN.
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
