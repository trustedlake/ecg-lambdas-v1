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
    from .signal_tinted import ink_evidence as measured_evidence
    raw,contrast=measured_evidence(a)
    n,labels,stats,_=cv2.connectedComponentsWithStats(raw,8)
    keep=(stats[:,cv2.CC_STAT_AREA]>=4);keep[0]=False
    keep[_page_edge_components(labels,stats)]=False
    raw=keep[labels].astype('uint8')
    return raw,contrast


def locate_leads(raw,config,geometry):
    from .signal_tinted import locate_leads as locate_joint
    return locate_joint(raw,config,geometry)


def trace_lead(raw,contrast,lead):
    from .signal_tinted import trace_lead as trace_measured
    return trace_measured(raw,contrast,lead)
