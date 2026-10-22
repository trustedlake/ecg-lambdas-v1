"""Experimental shared mask-to-curve reconstruction; no identity or lead rules.

Retain dominant centerline excursions instead of column medians. Multi-valued
columns remain uncertain. No resampling, interpolation or invented timing.
"""
import numpy as np
import cv2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def monotone(values):
    blocks=[]
    for i,v in enumerate(values):
        blocks.append([float(v),1,i,i+1])
        while len(blocks)>1 and blocks[-2][0]/blocks[-2][1]>blocks[-1][0]/blocks[-1][1]:
            b=blocks.pop(); a=blocks.pop(); blocks.append([a[0]+b[0],a[1]+b[1],a[2],b[3]])
    out=np.empty(len(values),float)
    for total,count,start,end in blocks: out[start:end]=total/count
    for run in np.split(np.arange(len(out)),np.flatnonzero(np.diff(out)!=0)+1):
        if len(run)>1:
            # At most one pixel of temporal uncertainty; no frequency claim.
            center=out[run[0]]
            left=.4 if run[0]==0 else min(.4,(center-out[run[0]-1])*.45)
            right=.4 if run[-1]==len(out)-1 else min(.4,(out[run[-1]+1]-center)*.45)
            out[run]=np.linspace(center-left,center+right,len(run))
    return out


def thin(mask):
    """Zhang-Suen thinning, with zero padding and no gap closing."""
    a = np.pad(np.asarray(mask, bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            n = sum(v.astype(np.uint8) for v in p)
            transitions = sum((~p[k] & p[(k+1) % 8]).astype(np.uint8) for k in range(8))
            if phase == 0:
                keep = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                keep = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = a[1:-1, 1:-1] & (n >= 2) & (n <= 6) & (transitions == 1) & keep
            if remove.any():
                a[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            return a[1:-1, 1:-1]


def reconstruct(trace, baseline_slope=0., rgb=None):
    """Follow centerline graph paths; retain gaps and timing uncertainty."""
    mask=np.asarray(trace['mask'],bool); x0,y0=trace['mask_origin'];h,w=mask.shape
    support=mask.copy()
    if rgb is not None:
        raw=np.asarray(rgb,np.float32)
        strength=(cv2.GaussianBlur(raw,(0,0),7)-raw).min(axis=2)[y0:y0+h,x0:x0+w]
        positive=strength[mask & (strength>0)]
        cutoff=max(3.,float(np.percentile(positive,50))*.15) if positive.size else 3.
        strong = mask & (strength>cutoff)
        count, component = cv2.connectedComponents(mask.astype(np.uint8),8)
        supported = np.bincount(component[strong],minlength=count)
        area = np.bincount(component[mask],minlength=count)
        accepted=(supported>=np.maximum(2,area*.5));accepted[0]=False
        support &= accepted[component]
    skel=thin(support); yy,xx=np.nonzero(skel); n=len(xx)
    if not n:
        x=np.asarray(trace['x_px'],float);y=np.full(len(x),np.nan);amb=np.ones(len(x),bool)
    else:
        ids=np.full(mask.shape,-1,np.int32);ids[yy,xx]=np.arange(n)
        rows=[];cols=[];cost=[]
        for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
            ny=yy+dy;nx=xx+dx;ok=(ny>=0)&(ny<h)&(nx>=0)&(nx<w)
            a=np.flatnonzero(ok);b=ids[ny[ok],nx[ok]];good=b>=0;a=a[good];b=b[good]
            rows.extend([a,b]);cols.extend([b,a]);cost.extend([np.full(len(a),np.hypot(dx,dy))]*2)
        graph=coo_matrix((np.concatenate(cost),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n)).tocsr()
        count,labels=connected_components(graph,directed=False)
        overlap=np.zeros(w,bool)
        for c in range(w):overlap[c]=len(np.unique(labels[xx==c]))>1
        segments=[]
        for label in range(count):
            nodes=np.flatnonzero(labels==label)
            if len(nodes)==1:
                segments.append((xx[nodes].astype(float),yy[nodes].astype(float),np.ones(1,bool)));continue
            left=nodes[xx[nodes]==xx[nodes].min()];right=nodes[xx[nodes]==xx[nodes].max()]
            start=int(left[len(left)//2]);end=int(right[len(right)//2])
            distance,parents=dijkstra(graph,indices=start,return_predecessors=True,directed=False)
            main=[end]
            while main[-1]!=start:main.append(int(parents[main[-1]]))
            main=main[::-1];on_main=set(main);branches={}
            valid_parents=parents[nodes];child_count=np.bincount(valid_parents[valid_parents>=0],minlength=n)
            for leaf in nodes[child_count[nodes]==0]:
                if leaf in on_main:continue
                chain=[int(leaf)]
                while chain[-1] not in on_main and parents[chain[-1]]>=0:chain.append(int(parents[chain[-1]]))
                anchor=chain[-1]
                if anchor not in on_main:continue
                excursion=float(np.max(np.abs(yy[chain]-yy[anchor])))
                if excursion<3:continue
                if anchor not in branches or excursion>branches[anchor][0]:branches[anchor]=(excursion,chain[::-1])
            path=[]
            for node in main:
                path.append(node)
                if node in branches:
                    branch=branches[node][1];path.extend(branch[1:]);path.extend(branch[-2::-1])
            path=np.asarray(path)
            actual=xx[path].astype(float)
            px=monotone(actual);py=yy[path].astype(float)
            # Timing below the printed stroke width is unresolved. Keep all
            # observed heights and explicitly flag the horizontal adjustment.
            shifted=np.abs(px-actual)
            radius=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)[yy[path],xx[path]]
            permitted=np.maximum(1.,radius*2.)
            rejected=shifted>permitted
            stroke_distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3)
            projected_x=np.clip(np.rint(px).astype(int),0,w-1)
            rejected |= stroke_distance[yy[path],projected_x]>1.
            py[rejected]=np.nan
            uncertain=np.asarray(trace['ambiguous'])[xx[path]] | overlap[xx[path]] | (shifted>.01)
            segments.append((px,py,uncertain))
            # Keep observed tips rejected solely by the monotone time fit.
            # Re-express that fragment at its original raster columns instead
            # of deleting its amplitude. Never synthesize additional pixels.
            rejected |= ~np.isfinite(py)
            if rejected.any():
                lost=path[rejected]
                rx=np.unique(xx[lost]).astype(float);ry=[]
                for col in rx.astype(int):
                    values=yy[lost[xx[lost]==col]]
                    base=trace['baseline_y_px']-y0+baseline_slope*(col-w/2)
                    ry.append(values[np.argmax(np.abs(values-base))])
                segments.append((rx,np.asarray(ry,float),np.ones(len(rx),bool)))
        # Resolve overlapping component intervals by observed excursion per
        # column, without letting a long baseline component erase a narrow tip.
        owner=np.full(w,-1,int);score=np.full(w,-np.inf)
        for index,(px,py,uncertain) in enumerate(segments):
            for col in np.unique(np.clip(np.rint(px).astype(int),0,w-1)):
                use=(np.rint(px).astype(int)==col)&np.isfinite(py)
                if not use.any():continue
                base=trace['baseline_y_px']-y0+baseline_slope*(col-w/2)
                value=float(np.max(np.abs(py[use]-base)))
                if value>score[col]:score[col]=value;owner[col]=index
        entries=[]
        for index,(px,py,uncertain) in enumerate(segments):
            for cx,cy,flag in zip(px,py,uncertain):
                col=int(np.clip(round(cx),0,w-1))
                if owner[col]==index:entries.append((cx,cy,bool(flag),index))
        entries.sort(key=lambda v:v[0]);xs=[];ys=[];flags=[];last=None
        for cx,cy,flag,index in entries:
            if last is not None and cx<=last[0]:continue
            if last is not None and (index!=last[1] or cx-last[0]>1.5):
                continuous=False
                if ys and np.isfinite(ys[-1]) and np.isfinite(cy):
                    steps=max(2,int(np.ceil(max(abs(cx-last[0]),abs(cy-ys[-1]))*2))+1)
                    test_x=np.clip(np.rint(np.linspace(last[0],cx,steps)).astype(int),0,w-1)
                    test_y=np.clip(np.rint(np.linspace(ys[-1],cy,steps)).astype(int),0,h-1)
                    continuous=bool(mask[test_y,test_x].all())
                if not continuous:
                    xs.append((last[0]+cx)/2);ys.append(np.nan);flags.append(True)
            xs.append(cx);ys.append(cy);flags.append(flag);last=(cx,index)
        x=np.asarray(xs)+x0;y=np.asarray(ys)+y0;amb=np.asarray(flags,bool)
    # A monotone time coordinate must not move evidence outside the printed
    # stroke. Reject unsupported samples rather than relaxing the evidence test.
    finite=np.flatnonzero(np.isfinite(y))
    if finite.size:
        distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3)
        cx=np.clip(np.rint(x[finite]-x0).astype(int),0,w-1)
        cy=np.clip(np.rint(y[finite]-y0).astype(int),0,h-1)
        unsupported=finite[distance[cy,cx]>1.]
        y[unsupported]=np.nan
        amb[unsupported]=True
    observed=np.isfinite(y)
    baseline=trace['baseline_y_px']+baseline_slope*(x-(x0+w/2))
    col=np.clip(np.rint(x-x0).astype(int),0,w-1)
    return dict(x_px=x,y_px=y,y_min_px=np.asarray(trace['y_min_px'])[col],
                y_max_px=np.asarray(trace['y_max_px'])[col],baseline_y_px=baseline,
                observed=observed,ambiguous=amb,valid=observed & ~amb)


