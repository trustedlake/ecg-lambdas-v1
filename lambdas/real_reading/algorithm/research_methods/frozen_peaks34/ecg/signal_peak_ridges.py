"""Opt-in narrow peak recovery after focal extraction.

Measured contrast only. Closing identifies fragments but never exports new
pixels. Existing lead ownership is protected. Competing ridges abstain at ties;
proximity to a ridge is heuristic, so every changed column stays ambiguous.
"""
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from .signal_neutral import summarize

def refine(image,leads,traces,names):
 report={}
 red=image[...,0].astype(float);f=image.astype(float)
 # dark ridge relative to horizontal surroundings; suppress long grid lines
 contrast=median_filter(red,size=(1,21))-red
 vertical=cv2.morphologyEx((contrast>15).astype('uint8'),cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
 tall=cv2.morphologyEx(vertical,cv2.MORPH_OPEN,np.ones((250,1),np.uint8))>0
 evidence=(contrast>18)&(f.max(2)<150)&~tall
 counts=np.zeros(red.shape,int);candidates={};costs={}
 for i,(l,t) in enumerate(zip(leads,traces)):
  if l['name'] not in names:continue
  x,y=t['mask_origin'];h,w=t['mask'].shape;step=l['row_spacing_px'];base=t['baseline_y_px']
  score=evidence[max(0,int(base-.85*step)):int(base-.2*step),x:x+w].sum(0)
  peaks,_=find_peaks(cv2.GaussianBlur(score.astype('float32')[None,:],(5,1),0).ravel(),prominence=8,distance=50)
  extra=np.zeros_like(t['mask']);cost=np.full(extra.shape,1e6)
  for peak in peaks:
   a=max(0,peak-7);b=min(w,peak+10)
   vals=t['y_px'][max(0,peak-25):max(1,peak-10)];baseline=float(np.nanmedian(vals)) if np.isfinite(vals).any() else base
   lo=max(y,int(baseline-.95*step));hi=min(y+h,int(baseline+.4*step))
   local=evidence[lo:hi,x+a:x+b].astype('uint8')
   if np.median(f[lo:hi,x+a:x+b].max(2).min(1))>105:continue
   graph=cv2.morphologyEx(local,cv2.MORPH_CLOSE,np.ones((7,3),np.uint8))
   n,labels,stats,_=cv2.connectedComponentsWithStats(graph,8)
   keep=np.zeros(n,bool)
   for k in range(1,n):
    xx,yy,ww,hh,area=stats[k];keep[k]=hh>step*.4 and yy+lo<baseline-step*.3 and yy+hh+lo>baseline-step*.15
   sel=keep[labels] & (local>0)
   extra[lo-y:hi-y,a:b] |= sel
   cost[lo-y:hi-y,a:b]=np.minimum(cost[lo-y:hi-y,a:b],np.where(sel,abs(np.arange(a,b)[None,:]-peak),1e6))
  candidates[i]=extra;costs[i]=cost;counts[y:y+h,x:x+w]+=extra
 best=np.full(red.shape,1e6);owner=np.full(red.shape,-1);second=best.copy()
 for i,extra in candidates.items():
  t=traces[i];x,y=t['mask_origin'];h,w=extra.shape;c=costs[i];b=best[y:y+h,x:x+w];sec=second[y:y+h,x:x+w];o=owner[y:y+h,x:x+w];better=c<b;sec[:]=np.where(better,b,np.minimum(sec,c));o[better]=i;b[:]=np.minimum(b,c)
 occupied=np.zeros(red.shape,bool)
 for l,t in zip(leads,traces):
  x,y=t['mask_origin'];h,w=t['mask'].shape
  if l['name'] not in names:occupied[y:y+h,x:x+w]|=t['mask']
 for i,extra in candidates.items():
  t=traces[i];x,y=t['mask_origin'];h,w=extra.shape
  # keep prior assigned target pixels; additions cannot steal them
  other=occupied.copy()
  for j,tt in enumerate(traces):
   if j==i:continue
   xx,yy=tt['mask_origin'];hh,ww=tt['mask'].shape;other[yy:yy+hh,xx:xx+ww]|=tt['mask']
  extra &= ~other[y:y+h,x:x+w] & (owner[y:y+h,x:x+w]==i)&(second[y:y+h,x:x+w]>best[y:y+h,x:x+w]+.5)
  changed=extra.any(0);report[leads[i]['name']]={'added_pixels':int((extra & ~t['mask']).sum()),'method':'narrow_measured_ridges_exclusive','limitation':'Nearest ridge is heuristic; changed columns uncertain, no interpolation.'};t['mask']|=extra;summarize(t,changed)
 return report
