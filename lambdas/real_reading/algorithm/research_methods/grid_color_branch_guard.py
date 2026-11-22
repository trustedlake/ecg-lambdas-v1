"""Experimental removal of isolated vertical branches with neighboring peak protection."""
import numpy as np
import cv2
from . import grid_color_periodic as previous, grid_color_paths as gp
from .grid_color_periodic import thin
def finite_median(values):
 values=np.asarray(values);values=values[np.isfinite(values)]
 return float(np.median(values)) if values.size else float('nan')

def prune(ts,leads):
 for t,l in zip(ts,leads):
  m=t['mask']; spans=np.nan_to_num(t['y_max_px']-t['y_min_px']); events=cv2.dilate((spans>l['row_spacing_px']*.5).astype(np.uint8).reshape(1,-1),np.ones((1,7),np.uint8)).ravel()>0
  if np.sum(events)<10: continue
  sk=thin(cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)))
  neighbors=cv2.filter2D(sk.astype(np.uint8),-1,np.ones((3,3),np.uint8))-sk
  endpoints=list(zip(*np.nonzero(sk&(neighbors==1))));remove=np.zeros_like(sk)
  for y,x in endpoints:
   route=[(y,x)];prev=None;cur=(y,x)
   for _ in range(int(l['row_spacing_px'])):
    yy,xx=cur
    ns=[(yy+dy,xx+dx) for dy in [-1,0,1] for dx in [-1,0,1] if (dy or dx) and 0<=yy+dy<sk.shape[0] and 0<=xx+dx<sk.shape[1] and sk[yy+dy,xx+dx] and (yy+dy,xx+dx)!=prev]
    if len(ns)!=1:break
    prev=cur;cur=ns[0];route.append(cur)
   yy,xx=np.array(route).T
   if len(route)>=6 and np.ptp(xx)<=2 and np.ptp(yy)>2*np.ptp(xx)+4:
    left=max(0,int(xx.min())-8);right=min(m.shape[1],int(xx.max())+9); spans=np.nan_to_num(t['y_max_px']-t['y_min_px']); surrounding=np.r_[spans[left:max(left,int(xx.min())-2)],spans[min(right,int(xx.max())+3):right]]
    nearby=np.r_[t['y_px'][left:max(left,int(xx.min())-2)],t['y_px'][min(right,int(xx.max())+3):right]]
    off_baseline=nearby.size and abs(finite_median(nearby)-t['baseline_y_px'])>l['row_spacing_px']*.18
    center=int(np.median(xx)); ys=t['y_px']; n=len(ys)
    li=finite_median(ys[max(0,center-7):max(0,center-2)]);lo=finite_median(ys[max(0,center-14):max(0,center-8)])
    ri=finite_median(ys[min(n,center+3):min(n,center+8)]);ro=finite_median(ys[min(n,center+9):min(n,center+15)])
    rounded=(lo-li>1) and (ro-ri>1) and abs(li-ri)<l['row_spacing_px']*.12
    if rounded and off_baseline and surrounding.size and surrounding.max()<l['row_spacing_px']*.35: remove[yy[:-2],xx[:-2]]=True
  retained=sk&~remove
  distkeep=cv2.distanceTransform((~retained).astype(np.uint8),cv2.DIST_L2,3)
  distdrop=cv2.distanceTransform((~remove).astype(np.uint8),cv2.DIST_L2,3)
  cut=m&(distdrop<distkeep)&(distdrop<3)
  gp.summarize(t,m&~cut,t['ambiguous']|cut.any(0))

def extract(context):
 result=previous.extract(context)
 prune(result.traces,result.leads)
 result.diagnostics['branch_filter']='isolated_vertical_branches_neighbor_peak_guard'
 return result
