"""Whole-image measured graph with layout used only to seed ownership.

No rectangular trace windows. Junctions pair by measured tangent continuity;
unresolved ownership remains unassigned. The reference is never a mask.
"""
import cv2
import numpy as np
from .signal_tinted import locate_leads


def thin(mask, max_stroke_radius=None):
    """Zhang-Suen thinning, retaining only original measured pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    result=np.zeros_like(mask,dtype=bool)
    for k in range(1,n):
        x,y,w,h,area=stats[k]
        # Process connected ink independently, so a thick object outside the
        # paper cannot force hundreds of full-photograph thinning passes.
        local=labels[y:y+h,x:x+w]==k
        if max_stroke_radius is not None and area/(w*h)>.2:
            radius=cv2.distanceTransform(np.pad(local.astype(np.uint8),1),cv2.DIST_L2,3)
            if float(radius.max())>max_stroke_radius:
                continue
        result[y:y+h,x:x+w] |= _thin_component(local)
    return result


def _thin_component(mask):
    a = np.pad(mask.astype(bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [a[:-2,1:-1], a[:-2,2:], a[1:-1,2:], a[2:,2:],
                 a[2:,1:-1], a[2:,:-2], a[1:-1,:-2], a[:-2,:-2]]
            count = sum(v.astype(np.uint8) for v in p)
            turns = sum((~p[k] & p[(k+1)%8]).astype(np.uint8) for k in range(8))
            if phase == 0:
                keep = (p[0]&p[2]&p[4]) | (p[2]&p[4]&p[6])
            else:
                keep = (p[0]&p[2]&p[6]) | (p[0]&p[4]&p[6])
            remove = a[1:-1,1:-1] & (count>=2) & (count<=6) & (turns==1) & ~keep
            if remove.any():
                a[1:-1,1:-1][remove] = False
                changed = True
        if not changed:
            return a[1:-1,1:-1]


def reference_leads(raw, config, geometry, contrast=None):
    x0,y0,x1,y1 = geometry['reference_box_px']
    local = dict(config, paper_baseline_fractions=config['baseline_fractions'])
    location=raw if contrast is None else raw*np.minimum(contrast,100)
    leads = locate_leads(location[y0:y1,x0:x1], local, {'method':'configured_quad'})
    for lead in leads:
        lead['baseline_y_px'] += y0
        lead['box'] = [lead['box'][0]+x0, lead['box'][1]+y0,
                       lead['box'][2]+x0, lead['box'][3]+y0]
        lead['trace_x'] = [v+x0 for v in lead['trace_x']]
        lead['time_origin_x_px'] = x0+(x1-x0)*lead['column']/len(
            [v for v in leads if v['row']==lead['row']])
    return leads


def remove_section_markers(raw, leads, reference):
    """Locate compact thick vertical print near expected section starts.

    Only the measured marker pixels are withheld. Its rectangle is not a
    domain limit, and paths may extend to either side through other pixels.
    """
    measured=raw.copy();markers=[]
    x0,_,x1,_=reference
    for row in sorted({lead['row'] for lead in leads}):
        group=[lead for lead in leads if lead['row']==row]
        if len(group)<2:continue
        step=group[0]['row_spacing_px'];pitch=(x1-x0)/len(group)
        boundaries=[x0]
        for col in range(1,len(group)):
            expected=x0+pitch*col;base=group[col]['baseline_y_px']
            a=max(0,int(expected-pitch*.16));b=min(raw.shape[1],int(expected+pitch*.16))
            top=max(0,int(base-step*.22));bottom=min(raw.shape[0],int(base+step*.22))
            local=raw[top:bottom,a:b]
            vertical=cv2.morphologyEx(local,cv2.MORPH_OPEN,
                                     np.ones((max(5,int(step*.07)),1),np.uint8))
            n,cc,stats,_=cv2.connectedComponentsWithStats(vertical,8)
            candidates=[]
            for k in range(1,n):
                xx,yy,w,h,area=stats[k]
                if (max(3,step*.008)<=w<=step*.055 and step*.07<=h<=step*.23
                    and area/(w*h)>.8 and yy+top<=base+step*.06
                    and yy+top+h>=base-step*.06):
                    candidates.append((abs(xx+a-expected),k))
            if candidates:
                _,k=min(candidates);xx,yy,w,h,_=stats[k]
                at=a+xx+w/2;boundaries.append(at)
                selected=cc==k
                measured[top:bottom,a:b][selected]=0
                markers.append([a+int(xx),top+int(yy),int(w),int(h)])
            else:boundaries.append(expected)
        boundaries.append(x1)
        for col,lead in enumerate(group):
            lead['trace_x']=[int(boundaries[col])+2,int(boundaries[col+1])-2]
            lead['time_origin_x_px']=float(boundaries[col])
    return measured,markers


def remove_calibration_pulses(raw, leads, reference):
    measured=raw.copy();pulses=[]
    left=reference[0]
    for lead in leads:
        if lead['column']!=0:continue
        step=lead['row_spacing_px'];base=lead['baseline_y_px']
        a=max(0,int(left-step*.5));b=min(raw.shape[1],int(left+step*.1))
        top=max(0,int(base-step*.5));bottom=min(raw.shape[0],int(base+step*.12))
        local=raw[top:bottom,a:b]
        vertical=cv2.morphologyEx(local,cv2.MORPH_OPEN,np.ones((max(5,int(step*.15)),1),np.uint8))
        n,cc,stats,_=cv2.connectedComponentsWithStats(vertical,8)
        found=False
        for i in range(1,n):
            x,y,w,h,_=stats[i]
            for j in range(i+1,n):
                xx,yy,ww,hh,_=stats[j]
                distance=abs((xx+ww/2)-(x+w/2))
                if not (.07*step<distance<.3*step and abs(yy-y)<.04*step
                        and abs(hh-h)<.08*step):continue
                xa=min(x,xx);xb=max(x+w,xx+ww)
                ya=max(0,min(y,yy)-2);yb=min(local.shape[0],max(y,yy)+4)
                plateau=local[ya:yb,xa:xb]
                if not plateau.size or plateau.any(0).mean()<.85:continue
                # Three observed sides of a rectangular calibration pulse.
                excluded=(cc==i)|(cc==j)
                excluded[ya:yb,xa:xb] |= plateau>0
                measured[top:bottom,a:b][excluded]=0
                pulses.append([a+int(xa),top+int(ya),int(xb-xa),int(max(y+h,yy+hh)-ya)])
                found=True;break
            if found:break
    return measured,pulses


def trace_reference(context, raw, contrast):
    from research_methods.contracts import Extraction
    raw = np.asarray(raw, dtype=np.uint8)
    leads = reference_leads(raw, context.config, context.geometry, contrast)
    strengths=[]
    for lead in leads:
        base=lead['baseline_y_px'];step=lead['row_spacing_px'];a,b=lead['trace_x']
        top=max(0,int(base-step*.1));bottom=min(raw.shape[0],int(base+step*.1)+1)
        local=contrast[top:bottom,a:b]*raw[top:bottom,a:b]
        if local.size:strengths.append(float(np.median(local.max(0))))
    threshold=.4*float(np.median(strengths)) if strengths else 0.
    strong=raw & (contrast>=threshold).astype(np.uint8)
    bracketed=np.zeros_like(raw)
    for kernel in (np.ones((7,1),np.uint8),np.ones((1,5),np.uint8),
                   np.eye(5,dtype=np.uint8),np.fliplr(np.eye(5,dtype=np.uint8))):
        bracketed |= cv2.morphologyEx(strong,cv2.MORPH_CLOSE,kernel)
    raw=strong | (raw & bracketed)
    raw,markers=remove_section_markers(raw,leads,context.geometry['reference_box_px'])
    raw,pulses=remove_calibration_pulses(raw,leads,context.geometry['reference_box_px'])
    skeleton = thin(raw > 0, max(6.,leads[0]['row_spacing_px']*.035))
    degree = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3,3),np.uint8))-skeleton
    junction = skeleton & (degree>2)
    # Collapse adjacent junction pixels into one node; edge tangents below are
    # measured beyond its thickness rather than inferred from one noisy pixel.
    junction_area = (cv2.dilate(junction.astype(np.uint8), np.ones((3,3),np.uint8))>0)&skeleton
    count, segments, segment_stats, _ = cv2.connectedComponentsWithStats((skeleton & ~junction_area).astype(np.uint8), 8)
    n_nodes, nodes, node_stats, node_centers = cv2.connectedComponentsWithStats(junction_area.astype(np.uint8), 8)
    parent = np.arange(count)
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b: parent[b]=a
    neighborhoods=[]
    for k in range(1,n_nodes):
        nx,ny,nw,nh,_=node_stats[k]
        xa,xb=max(0,nx-2),min(raw.shape[1],nx+nw+2)
        ya,yb=max(0,ny-2),min(raw.shape[0],ny+nh+2)
        touching=cv2.dilate((nodes[ya:yb,xa:xb]==k).astype(np.uint8),np.ones((3,3),np.uint8))>0
        ids=np.unique(segments[ya:yb,xa:xb][touching]); ids=ids[ids>0]
        center=node_centers[k]
        directions={}
        for sid in ids:
            ex,ey,ew,eh,_=segment_stats[sid]
            sy,sx=np.where(segments[ey:ey+eh,ex:ex+ew]==sid)
            sy+=ey; sx+=ex
            pts=np.column_stack([sx,sy]); distance=np.linalg.norm(pts-center,axis=1)
            pts=pts[np.argsort(distance)[:12]]
            vector=pts.mean(0)-center
            norm=np.linalg.norm(vector)
            if norm: directions[int(sid)]=vector/norm
        pairs=sorted((float(np.dot(directions[a],directions[b])),a,b)
                     for a in directions for b in directions if a<b)
        used=set()
        for dot,a,b in pairs:
            if dot < -.85 and a not in used and b not in used:
                union(a,b); used.update((a,b))
        neighborhoods.append((k,ids.tolist()))
    roots=np.array([root(i) for i in range(count)])
    votes=np.zeros((count,len(leads)),np.int32)
    ys,xs=np.where(skeleton & ~junction_area)
    horizontal_seed=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((1,5),np.uint8))>0
    for i,lead in enumerate(leads):
        a,b=lead['trace_x']; base=lead['baseline_y_px']; step=lead['row_spacing_px']
        seed=(xs>=a)&(xs<b)&(np.abs(ys-base)<max(2,.055*step))&horizontal_seed[ys,xs]
        votes[:,i]=np.bincount(roots[segments[ys[seed],xs[seed]]], minlength=count)
    owner=np.full(count,-1,np.int16)
    for sid in np.unique(roots[1:]):
        evidence=votes[sid]; candidates=np.flatnonzero(evidence>=3)
        if len(candidates)==1: owner[sid]=candidates[0]
    # A dangling excursion can attach to a uniquely owned node. At crossings
    # between origins this propagation is deliberately forbidden.
    for _ in range(count):
        changed=False
        for _,ids in neighborhoods:
            rr=np.unique(roots[ids]); known=np.unique(owner[rr]); known=known[known>=0]
            if len(known)==1 and not any(np.count_nonzero(votes[r]>=3)>1 for r in rr):
                for r in rr:
                    if owner[r]<0: owner[r]=known[0]; changed=True
        if not changed: break
    assigned=owner[roots[segments]]; assigned[segments==0]=-1
    for k,ids in neighborhoods:
        known=np.unique(owner[roots[ids]]);known=known[known>=0]
        if len(known)==1:
            nx,ny,nw,nh,_=node_stats[k]
            selected=(nodes[ny:ny+nh,nx:nx+nw]==k)&skeleton[ny:ny+nh,nx:nx+nw]
            assigned[ny:ny+nh,nx:nx+nw][selected]=known[0]
    traces=[]
    for i,lead in enumerate(leads):
        mask=(assigned==i)&skeleton
        yy,xx=np.where(mask)
        # The domain is measured ink, not the reference or a section rectangle.
        left=int(xx.min()) if len(xx) else int(lead['trace_x'][0])
        right=int(xx.max())+1 if len(xx) else int(lead['trace_x'][1])
        selected=mask[:,left:right]; width=right-left
        y=np.full(width,np.nan,np.float32);low=y.copy();high=y.copy()
        ambiguity=np.zeros(width,bool)
        base=lead['baseline_y_px']
        for x in np.flatnonzero(selected.any(0)):
            points=np.flatnonzero(selected[:,x]);low[x]=points[0];high[x]=points[-1]
            # Preserve a measured extremum, never a median floating between ink.
            y[x]=points[np.argmax(np.abs(points-base))]
            ambiguity[x]=len(points)>1
        observed=selected.any(0);valid=observed&~ambiguity
        traces.append(dict(mask=selected,mask_origin=[left,0],
            x_px=np.arange(left,right,dtype=np.float32),y_px=y,
            y_min_px=low,y_max_px=high,observed=observed,valid=valid,ambiguous=ambiguity,
            baseline_y_px=base,time_origin_x_px=lead['time_origin_x_px'],
            coverage=float(observed.mean()),unambiguous_fraction=float(valid.mean()),
            method='reference_seeded_measured_graph',baseline_method='layout_geometric_only'))
    return Extraction(leads,traces,{'unassigned_skeleton_pixels':int((skeleton&(assigned<0)).sum()),
                      'reference_box_px':context.geometry['reference_box_px'],
                      'measured_markers':markers,
                      'measured_calibration_pulses':pulses,
                      'measured_trace_contrast_threshold':threshold,
                      'junctions':n_nodes-1,'segments':count-1})
