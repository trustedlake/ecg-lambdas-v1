"""Opt-in measured faint ink on a locally sheared red grid.

Long horizontal/vertical background structures are discounted in the red
channel. Only measured residuals are exported; grid crossings and long flat
strokes may remain incomplete. Geometry and calibration are not changed.
"""
import cv2
import numpy as np
from scipy.ndimage import median_filter
from .signal_tinted import locate_leads, trace_lead, recover_stroke_pixels


def ink_evidence(image):
    red = image[..., 0].astype(np.float32)
    background = np.minimum(median_filter(red, size=(31, 1)),
                            median_filter(red, size=(1, 31)))
    contrast = np.maximum(0, background - red)
    raw = ((contrast > 6) & (red < 242)).astype('uint8')
    _, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 4
    keep[0] = False
    return keep[labels].astype('uint8'), contrast


def stroke_evidence(image):
    return ink_evidence(image)


def refine_peaks(image, leads, traces, names, avoid_owned=False):
 """Recover measured narrow dark peak branches; abstain at occupied/tied pixels.

 Internal closing connects fragments only. Every changed column is ambiguous.
 """
 from scipy.signal import find_peaks
 from .trace_refinement import _recompute
 red=image[...,0].astype(np.float32)
 horizontal_bg=median_filter(red,size=(1,31))
 measured=(horizontal_bg-red>9)&(red<200)
 occupied=np.zeros(image.shape[:2],np.int16)
 candidates={}
 for i,t in enumerate(traces):
  x,y=t['mask_origin'];h,w=t['mask'].shape;occupied[y:y+h,x:x+w][t['mask']]=i+1
 for i,(l,t) in enumerate(zip(leads,traces)):
  if l['name'] not in names:continue
  x,y=t['mask_origin'];h,w=t['mask'].shape;base=t['baseline_y_px'];step=l['row_spacing_px']
  lo=max(y,int(base-step*(1.4 if avoid_owned else 1.05)));hi=int(base-step*.18)
  score=np.maximum(0,180-red[lo:hi,x:x+w]).mean(0)
  peaks,_=find_peaks(score,prominence=3,distance=int(step*.45))
  extra=np.zeros_like(t['mask'])
  for peak in peaks:
   a=max(0,peak-7);b=min(w,peak+10)
   local_base=base
   if avoid_owned:
    nearby=t['y_px'][max(0,peak-30):max(1,peak-10)]
    if np.isfinite(nearby).any():local_base=float(np.nanmedian(nearby))
   if not t['mask'][max(0,int(local_base-y-12)):int(local_base-y+8),a:b].any():continue
   if not t['mask'][max(0,int(local_base-y+4)):int(local_base-y+20),a:b].any():continue
   roi=red[lo:int(local_base+step*.32),x+a:x+b]
   if (roi<160).sum()<12:continue
   # Measured narrow branch, starting with its dark center; no synthetic export.
   cost=roi.copy()
   if avoid_owned:
    owners=occupied[lo:lo+len(roi),x+a:x+b]
    cost[(owners!=0)&(owners!=i+1)]=1000
   path=np.argmin(cost,axis=1)
   for row,col in enumerate(path):
    gy=lo+row;gx=x+a+col
    if roi[row,col]>215:continue
    for dx in range(-2,3):
     xx=gx+dx
     if x+a<=xx<x+b and measured[gy,xx] and occupied[gy,xx] in (0,i+1):extra[gy-y,xx-x]=True
  graph=cv2.morphologyEx(extra.astype('uint8'),cv2.MORPH_CLOSE,np.ones((11,3),np.uint8))
  n,labels,stats,_=cv2.connectedComponentsWithStats(graph,8)
  near=t['mask'].copy();near[:max(0,int(base-y-step*.28))]=False
  touched=np.unique(labels[cv2.dilate(near.astype('uint8'),np.ones((5,5),np.uint8))>0])
  extra &= np.isin(labels,touched[touched!=0])
  candidates[i]=extra
 counts=np.zeros(image.shape[:2],np.uint16)
 for i,extra in candidates.items():
  x,y=traces[i]['mask_origin'];h,w=extra.shape;counts[y:y+h,x:x+w]+=extra
 report={}
 for i,extra in candidates.items():
  t=traces[i];x,y=t['mask_origin'];h,w=extra.shape
  extra &= counts[y:y+h,x:x+w]==1
  report[leads[i]['name']]={'added_pixels':int((extra & ~t['mask']).sum())}
  _recompute(t,t['mask']|extra)
 return report
