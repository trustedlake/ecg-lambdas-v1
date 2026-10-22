"""Experimental optical-density separation of reddish grid and neutral ink.
Identity-free specialist; no registry/production integration.
"""
import cv2
import numpy as np
from .contracts import Extraction, MethodAbstention
from .frozen_peaks34.ecg import signal_tinted
from . import faint_ink_ownership as trace_ownership
VERSION='grid-color-stroke-contrast'

def evidence(image):
    f=image.astype(np.float32)
    paper=np.maximum(np.percentile(f,95,axis=(0,1)),1)
    density=-np.log(np.clip((f+1)/(paper+1),.004,1))
    red=(f[...,0]>f[...,1]+15)&(f[...,0]>f[...,2]+5)&(f[...,0]>120)&(f[...,1]<235)
    fraction=float(red.mean())
    if fraction<.015:
        raise MethodAbstention('chromatic_residual: insufficient reddish grid evidence')
    # Grid absorption ratio is measured from reddish pixels, not source identity.
    ratio=float(np.clip(np.median(density[...,0][red]/np.maximum(density[...,1][red],.02)),.05,.65))
    residual=density[...,0]-ratio*density[...,1]
    from scipy.ndimage import percentile_filter
    lines=cv2.HoughLinesP(red.astype(np.uint8),1,np.pi/1800,80,
                          minLineLength=min(image.shape[:2])*.3,maxLineGap=15)
    angles=[]
    if lines is not None:
        for x0,y0,x1,y1 in lines.reshape(-1,4):
            dx,dy=float(x1-x0),float(y1-y0)
            if abs(dx)<abs(dy)*.1: angles.append(-np.degrees(np.arctan(dx/dy)))
    angle=float(np.median(angles)) if angles else 0.
    transform=cv2.getRotationMatrix2D((image.shape[1]/2,image.shape[0]/2),angle,1)
    red_channel=cv2.warpAffine(f[...,0],transform,(image.shape[1],image.shape[0]),borderMode=cv2.BORDER_REFLECT)

    # Upper quantile tolerates repeated aligned dark peaks; a median does not.
    length=max(61,int(image.shape[0]*.25)|1)
    vertical=percentile_filter(red_channel,80,size=(length,1),mode='nearest')
    horizontal=percentile_filter(red_channel,80,size=(1,61),mode='nearest')
    background=cv2.warpAffine(np.minimum(vertical,horizontal),cv2.invertAffineTransform(transform),
                              (image.shape[1],image.shape[0]),borderMode=cv2.BORDER_REFLECT)
    strength=np.maximum(0,background-f[...,0])
    # Sustained reddish support identifies grid independently from dark peaks.
    support=cv2.blur(red.astype(np.float32),(3,max(31,int(image.shape[0]*.08)|1)))
    grid=support>.32
    raw=((strength>18)&(~grid|(strength>40))).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(raw,8)
    keep=stats[:,cv2.CC_STAT_AREA]>=4;keep[0]=False
    return keep[labels].astype(np.uint8),strength,{'red_fraction':fraction,'grid_absorption_ratio':ratio,'grid_angle':angle}

def extract(context):
    raw,strength,metrics=evidence(context.image)
    leads=signal_tinted.locate_leads(raw,context.config,context.geometry)
    traces=[]
    for lead in leads:
        trace=signal_tinted.trace_lead(raw,strength,lead)
        trace=signal_tinted.recover_stroke_pixels(raw>0,trace,lead)
        traces.append(trace)
    ownership=trace_ownership.separate_traces(traces,leads,context.image.shape)
    return Extraction(leads,traces,{'activation':metrics,'ownership':ownership})

