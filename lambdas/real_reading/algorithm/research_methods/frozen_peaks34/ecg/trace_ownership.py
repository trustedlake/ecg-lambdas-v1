"""Joint, exclusive ownership of measured pixels for the opt-in trace profile."""
import heapq
import math
import cv2
import numpy as np


def _distances(graph, seeds):
    """Geodesic distance along supported strokes, not distance to a row box."""
    h,w=graph.shape
    distance=np.full((h,w),np.inf,np.float32)
    yy,xx=np.nonzero(seeds & graph)
    distance[yy,xx]=0
    queue=[(0.,int(y),int(x)) for y,x in zip(yy,xx)]
    heapq.heapify(queue)
    neighbours=[(dy,dx,math.hypot(dy,dx)) for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx]
    while queue:
        value,y,x=heapq.heappop(queue)
        if value>float(distance[y,x])+1e-5:continue
        for dy,dx,cost in neighbours:
            ny,nx=y+dy,x+dx
            if 0<=ny<h and 0<=nx<w and graph[ny,nx]:
                candidate=value+cost
                if candidate<float(distance[ny,nx])-1e-5:
                    distance[ny,nx]=candidate
                    heapq.heappush(queue,(float(distance[ny,nx]),ny,nx))
    return distance


def _anchors(trace, lead):
    mask=trace['mask'];x0,y0=trace['mask_origin'];step=lead['row_spacing_px']
    rows=np.arange(mask.shape[0])+y0
    core=mask & (abs(rows[:,None]-trace['baseline_y_px'])<=.25*step)
    # A neighbour's narrow vertical peak crossing this band is not an anchor.
    horizontal=cv2.morphologyEx(core.astype('uint8'),cv2.MORPH_OPEN,np.ones((1,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(horizontal,8)
    keep=np.zeros(count,bool)
    keep[1:]=stats[1:,cv2.CC_STAT_WIDTH]>=5
    return keep[labels]


def separate_traces(traces, leads, image_shape):
    """Keep a pixel only for its uniquely supported owner; abstain at ties.

    Small gaps are bridged solely for connectivity. Exported pixels are always
    a subset of each original measured mask. No presentation-box clipping.
    """
    counts=np.zeros(image_shape[:2],np.uint16)
    for trace in traces:
        x,y=trace['mask_origin'];h,w=trace['mask'].shape
        counts[y:y+h,x:x+w]+=trace['mask']
    graph=cv2.morphologyEx((counts>0).astype('uint8'),cv2.MORPH_CLOSE,np.ones((7,1),np.uint8))
    graph=cv2.morphologyEx(graph,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
    best=np.full(graph.shape,np.inf,np.float32);second=best.copy()
    owner=np.full(graph.shape,-1,np.int16)
    for index,(trace,lead) in enumerate(zip(traces,leads)):
        x,y=trace['mask_origin'];h,w=trace['mask'].shape
        region=np.s_[y:y+h,x:x+w]
        distance=_distances(graph[region],_anchors(trace,lead))
        current=best[region];runner=second[region];labels=owner[region]
        better=distance<current
        runner[:]=np.where(better,current,np.minimum(runner,distance))
        labels[better]=index
        current[:]=np.minimum(current,distance)
    # A nearly equal route from two derivations cannot establish ownership.
    finite=np.isfinite(best)
    # A shorter route alone cannot disambiguate superimposed waveform strokes.
    # Include the anti-aliased fringe so different edges of one disputed stroke
    # do not remain painted in competing colours.
    disputed=cv2.dilate((counts>1).astype('uint8'),np.ones((3,3),np.uint8))>0
    unique=finite & (second>best+2.) & ~disputed
    owner[~unique]=-1
    report={'method':'exclusive_geodesic_anchors_abstain_at_conflicts',
            'shared_pixels_before':int((counts>1).sum()),'leads':{}}
    report['disputed_measured_pixels']=int((disputed & (counts>0)).sum())
    after=np.zeros_like(counts)
    for index,(trace,lead) in enumerate(zip(traces,leads)):
        x,y0=trace['mask_origin'];h,w=trace['mask'].shape
        selected=trace['mask'] & (owner[y0:y0+h,x:x+w]==index)
        removed=trace['mask'] & ~selected
        ambiguous=trace['ambiguous'] | removed.any(0)
        observed=selected.any(0)
        center=np.full(w,np.nan,np.float32);low=center.copy();high=center.copy()
        for col in np.flatnonzero(observed):
            points=np.flatnonzero(selected[:,col])+y0
            center[col]=np.median(points);low[col]=points[0];high[col]=points[-1]
        valid=observed & ~ambiguous
        trace.update(mask=selected,y_px=center,y_min_px=low,y_max_px=high,
            observed=observed,ambiguous=ambiguous,valid=valid,
            baseline_y_px=float(np.nanmedian(center)) if observed.any() else trace['baseline_y_px'],
            coverage=float(observed.mean()),unambiguous_fraction=float(valid.mean()))
        after[y0:y0+h,x:x+w]+=selected
        report['leads'][lead['name']]={'removed_pixels':int(removed.sum()),
            'affected_columns':int(removed.any(0).sum())}
    report['shared_pixels_after']=int((after>1).sum())
    if report['shared_pixels_after']:raise AssertionError('Non-exclusive trace ownership')
    return report
