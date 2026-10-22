"""Opt-in paths for dark ink on a chromatically separated regular grid.

No source names, prior waveforms or fixed image geometry enter extraction.
Unsupported route pixels are withheld; steep raster strokes remain ambiguous.
"""
import numpy as np
from scipy.ndimage import label, binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .signal import locate_leads


def ink_ratio(a):
    light=a.astype(np.float64).max(axis=2)
    col=np.percentile(light,85,axis=0)
    row=np.percentile(light,85,axis=1)
    background=row[:,None]*col[None,:]/max(float(np.percentile(light,95)),1)
    return light/np.maximum(background,1)


def make_graph(weights):
    h,w=weights.shape; ids=np.arange(h*w).reshape(h,w)
    fr=[]; to=[]; ws=[]
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        sy=slice(0,h-dy); ty=slice(dy,h)
        sx=slice(max(0,-dx),min(w,w-dx)); tx=slice(max(0,dx),min(w,w+dx))
        u=ids[sy,sx].ravel(); v=ids[ty,tx].ravel()
        cost=((weights[sy,sx]+weights[ty,tx])/2*np.hypot(dx,dy)).ravel()
        fr.extend([u,v]); to.extend([v,u]); ws.extend([cost,cost])
    return coo_matrix((np.concatenate(ws),(np.concatenate(fr),np.concatenate(to))),shape=(h*w,h*w)).tocsr()


def route(graph,start,end,w):
    _,previous=dijkstra(graph,indices=start,return_predecessors=True)
    path=[]; p=end
    while p!=start:
        path.append(p); p=int(previous[p])
        if p<0: raise RuntimeError('Disconnected trace')
    path.append(start)
    ys,xs=np.divmod(path[::-1],w)
    return np.column_stack([xs,ys])


def extract_lead(r,base_local,step):
    h,w=r.shape
    weights=np.exp(13*np.clip(r,0,1.3))
    graph=make_graph(weights)
    def endpoint(x):
        yy=np.arange(max(0,base_local-int(step*.08)),min(h,base_local+int(step*.08)+1))
        return int(yy[np.argmin(weights[yy,x])])
    initial=route(graph,endpoint(0)*w,endpoint(w-1)*w+w-1,w)
    base_local=float(np.median(initial[:,1]))
    seeds=np.zeros((h,w),bool); seeds[initial[:,1],initial[:,0]]=True
    mask=r<.68
    components,_=label(mask,np.ones((3,3)))
    touched=np.unique(components[binary_dilation(seeds,iterations=1)])
    mask &= np.isin(components,touched[touched!=0])
    anchors=[]
    for sign in [-1,1]:
        excursion=(np.arange(h)[:,None]-base_local)*sign
        large=mask & (excursion>max(4,step*.055))
        groups,ng=label(large.any(axis=0))
        for k in range(1,ng+1):
            xx=np.flatnonzero(groups==k)
            if len(xx)>max(6,step*.13): continue
            yy,xx2=np.nonzero(large[:,xx])
            if len(yy)<8: continue
            best=np.argmax((yy-base_local)*sign)
            ay=int(yy[best]); ax=int(xx[xx2[best]])
            if mask[max(0,ay-2):ay+3,max(0,ax-2):ax+3].sum()<4: continue
            anchors.append((ax,ay))
    anchors=[tuple(initial[0])]+sorted(set(anchors))+[tuple(initial[-1])]
    paths=[np.array([anchors[0]])]
    for (ax,ay),(bx,by) in zip(anchors[:-1],anchors[1:]):
        paths.append(route(graph,ay*w+ax,by*w+bx,w)[1:])
    return np.concatenate(paths),np.array(anchors)



def summarize(mask, origin, base, suspect=None):
    x0,y0=origin; width=mask.shape[1]
    y=np.full(width,np.nan,np.float32); low=y.copy(); high=y.copy()
    observed=mask.any(0); ambiguous=~observed
    for x in np.flatnonzero(observed):
        points=np.flatnonzero(mask[:,x])+y0
        low[x],high[x]=points[0],points[-1]
        steep=high[x]-low[x]>6
        y[x]=points[np.argmax(abs(points-base))] if steep else points[np.argmin(abs(points-np.median(points)))]
        ambiguous[x]=steep or np.any(np.diff(points)>1)
    if suspect is not None: ambiguous |= suspect
    valid=observed & ~ambiguous
    return dict(x_px=np.arange(x0,x0+width,dtype=np.float32),y_px=y,y_min_px=low,y_max_px=high,
        mask=mask,mask_origin=[x0,y0],observed=observed,ambiguous=ambiguous,valid=valid,
        baseline_y_px=float(np.nanmedian(y)) if observed.any() else float(base),
        baseline_method='median_observed_trace_not_clinical_isoelectric_reference',
        method='chromatic_ratio_extremum_paths',coverage=float(observed.mean()),unambiguous_fraction=float(valid.mean()))


def extract(a,grid,config,geometry):
    ratio=ink_ratio(a)
    leads=locate_leads((ratio<.68).astype('uint8'),config,geometry)
    traces=[]
    for lead in leads:
        x0,x1=lead['trace_x']; base=lead['baseline_y_px'];step=lead['row_spacing_px']
        y0=max(0,int(base-step*.65));y1=min(len(a),int(base+step*.70))
        r=ratio[y0:y1,x0:x1].copy()
        # Text below the beginning of a segment can touch a real QRS.
        # Penalize this region for routing and mark any exported points for review.
        yy,xx=np.indices(r.shape)
        text=(xx<step*.23)&(yy+y0>base+step*.10)&(yy+y0<base+step*.25)
        r[text]=np.maximum(r[text],.78)
        path,anchors=extract_lead(r,base-y0,step)
        measured=ratio[path[:,1]+y0,path[:,0]+x0]<.80
        mask=np.zeros(r.shape,bool);p=path[measured];mask[p[:,1],p[:,0]]=True
        suspect=(mask & ((ratio[y0:y1,x0:x1]>.68)|text)).any(0)
        traces.append(summarize(mask,[x0,y0],base,suspect))
    # Do not give overlapping measured pixels to different derivations.
    counts=np.zeros(a.shape[:2],np.uint16)
    for t in traces:
        x,y=t['mask_origin'];h,w=t['mask'].shape;counts[y:y+h,x:x+w]+=t['mask']
    shared=counts>1
    for i,t in enumerate(traces):
        x,y=t['mask_origin'];h,w=t['mask'].shape;dispute=shared[y:y+h,x:x+w]
        traces[i]=summarize(t['mask'] & ~dispute,[x,y],t['baseline_y_px'],t['ambiguous']|dispute.any(0))
    return leads,traces,dict(method='chromatic_ratio_paths_joint_abstention',shared_pixels_before=int(shared.sum()),shared_pixels_after=0,
        limitation='Raster extrema and text crossings remain ambiguous; no reference waveform used.')
