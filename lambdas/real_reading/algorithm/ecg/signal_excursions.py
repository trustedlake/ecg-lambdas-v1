"""Opt-in measured peak branches across row boxes; preserve owners and mark additions uncertain."""
import numpy as np
import cv2
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from .signal_neutral import summarize


def evidence(image,connectivity=False):
    red=image[...,0].astype('float32')
    contrast=median_filter(red,size=(1,31))-red
    core=(contrast>18)&(red<155)
    near=cv2.dilate(core.astype('uint8'),np.ones((3,3),np.uint8))>0
    raw=(contrast>12)&(red<215)
    connected=cv2.morphologyEx(raw.astype('uint8'),cv2.MORPH_CLOSE,np.ones((9,1),np.uint8))
    tall=cv2.morphologyEx(connected,cv2.MORPH_OPEN,np.ones((280,1),np.uint8))>0
    return raw & ~tall & (True if connectivity else near)

def refine(image,leads,traces,directions):
    raw=evidence(image,connectivity=True);measured=evidence(image);candidates={};costs={};report={}
    for i,(l,t) in enumerate(zip(leads,traces)):
        if l['name'] not in directions:continue
        direction=-1 if directions[l['name']]=='up' else 1;x,y=t['mask_origin'];h,w=t['mask'].shape
        base=t['baseline_y_px'];step=l['row_spacing_px']
        low=max(0,int(base-(.65 if direction<0 else -.18)*step));high=min(len(raw),int(base+(.65 if direction>0 else -.18)*step))
        if high<=low or w==0:continue
        vertical=cv2.morphologyEx(raw[low:high,x:x+w].astype('uint8'),cv2.MORPH_OPEN,np.ones((5,1),np.uint8))
        score=vertical.sum(0).astype('float32')
        peaks,_=find_peaks(cv2.GaussianBlur(score[None,:],(5,1),0).ravel(),prominence=4,distance=int(step*.45))
        extra=np.zeros_like(t['mask']);cost=np.full(extra.shape,1e6)
        for peak in peaks:
            vals=t['y_px'][max(0,peak-25):max(1,peak-10)]
            b=float(np.nanmedian(vals)) if np.isfinite(vals).any() else base
            a=max(0,peak-8);z=min(w,peak+10)
            lo=max(y,int(b-(1.18 if direction<0 else .20)*step));hi=min(y+h,int(b+(1.0 if direction>0 else .25)*step))
            if hi<=lo:continue
            local=raw[lo:hi,x+a:x+z].astype('uint8')
            graph=cv2.morphologyEx(local,cv2.MORPH_CLOSE,np.ones((9,3),np.uint8))
            n,cc,stats,_=cv2.connectedComponentsWithStats(graph,8);keep=np.zeros(n,bool)
            for k in range(1,n):
                xx,yy,ww,hh,area=stats[k]
                near=yy+lo<b+step*.13 and yy+hh+lo>b-step*.13
                extension=(b-(yy+lo)) if direction<0 else (yy+hh+lo-b)
                keep[k]=near and extension>step*.24 and hh>step*.28
            sel=keep[cc] & measured[lo:hi,x+a:x+z]
            # Isolated dark grid crossings are not a peak branch. Connectivity
            # joins short raster gaps only; every exported pixel stays measured.
            joined=cv2.morphologyEx(sel.astype('uint8'),cv2.MORPH_CLOSE,np.ones((7,1),np.uint8))
            count,parts,part_stats,_=cv2.connectedComponentsWithStats(joined,8)
            good=np.zeros(count,bool)
            for k in range(1,count):
                xx,yy,ww,hh,area=part_stats[k]
                good[k]=hh>=max(9,step*.12) and int((sel & (parts==k)).sum())>=15
            sel &= good[parts]
            extra[lo-y:hi-y,a:z] |= sel
            cost[lo-y:hi-y,a:z]=np.minimum(cost[lo-y:hi-y,a:z],np.where(sel,abs(np.arange(a,z)[None,:]-peak),1e6))
        candidates[i]=extra;costs[i]=cost
    best=np.full(raw.shape,1e6);second=best.copy();owner=np.full(raw.shape,-1)
    occupied=np.zeros(raw.shape,np.int16)
    for i,t in enumerate(traces):
        x,y=t['mask_origin'];h,w=t['mask'].shape;occupied[y:y+h,x:x+w][t['mask']]=i+1
    for i,extra in candidates.items():
        x,y=traces[i]['mask_origin'];h,w=extra.shape;b=best[y:y+h,x:x+w];s=second[y:y+h,x:x+w];o=owner[y:y+h,x:x+w];c=costs[i];better=c<b
        s[:]=np.where(better,b,np.minimum(s,c));o[better]=i;b[:]=np.minimum(b,c)
    for i,extra in candidates.items():
        t=traces[i];x,y=t['mask_origin'];h,w=extra.shape;occ=occupied[y:y+h,x:x+w]
        extra &= ((occ==0)|(occ==i+1)) & (owner[y:y+h,x:x+w]==i) & (second[y:y+h,x:x+w]>best[y:y+h,x:x+w]+.5)
        added=extra & ~t['mask'];t['mask']|=extra;summarize(t,added.any(0))
        report[leads[i]['name']]={'added_pixels':int(added.sum())}
    return report

