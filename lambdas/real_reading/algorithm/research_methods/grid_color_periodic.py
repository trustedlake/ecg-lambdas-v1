"""Experimental grid-period background and wide measured-stroke support.
Independent specialist candidate. No production activation or image identity rules.
"""
from copy import deepcopy
import numpy as np
import cv2
from scipy.signal import find_peaks
from .contracts import Extraction,MethodAbstention
from . import chromatic_residual as previous, grid_color_ownership as own, grid_color_paths as gp
from .grid_color_wide_trace import trace_lead
from .frozen_peaks34.ecg import signal_tinted as st

def evidence(rgb):
 f=rgb.astype(np.float32);chroma=f[:,:,0]-f[:,:,1]
 lines=cv2.HoughLinesP((chroma>18).astype(np.uint8),1,np.pi/1800,50,minLineLength=150,maxLineGap=10)
 hs=[];vs=[]
 for x0,y0,x1,y1 in ([] if lines is None else lines.reshape(-1,4)):
  dx=x1-x0;dy=y1-y0
  if abs(dx)>10*abs(dy):hs.append(dy/dx)
  if abs(dy)>10*abs(dx):vs.append(dx/dy)
 if not hs or not vs: raise MethodAbstention('No two-axis grid')
 mat=np.array([[1,-np.median(vs),0],[-np.median(hs),1,0]],np.float32)
 align=cv2.warpAffine(f,mat,(rgb.shape[1],rgb.shape[0]),borderValue=(255,255,255))
 chroma=align[:,:,0]-align[:,:,1]
 periods=[]
 for axis in [0,1]:
  profile=np.mean(chroma,axis=axis);profile-=cv2.GaussianBlur(profile.reshape(1,-1),(0,0),30).ravel()
  ac=np.correlate(profile,profile,'full')[len(profile)-1:];ac/=ac[0]
  peaks,_=find_peaks(ac[4:100]);peaks+=4
  if not len(peaks) or ac[peaks].max()<.3: raise MethodAbstention('Weak periodic grid')
  best=peaks[ac[peaks]>.85*ac[peaks].max()][0];xs=np.arange(best-1,best+2);ys=ac[xs];q=np.polyfit(xs,ys,2);periods.append(float(-q[1]/(2*q[0])))
 
 backgrounds=[]
 for axis,period in enumerate(periods):
  stack=[]
  for n in [-3,-2,-1,1,2,3]:
   tr=np.array([[1,0,n*period if axis==0 else 0],[0,1,n*period if axis==1 else 0]],np.float32)
   stack.append(cv2.warpAffine(align,tr,(rgb.shape[1],rgb.shape[0]),borderMode=cv2.BORDER_REFLECT))
  backgrounds.append(np.percentile(np.stack(stack),70,axis=0).astype(np.float32))
 # Different direction copies protect both horizontal and vertical waveform portions.
 bg=np.minimum(backgrounds[0],backgrounds[1]);res=np.maximum(0,bg-align)
 e=cv2.warpAffine(res[:,:,0],cv2.invertAffineTransform(mat),(rgb.shape[1],rgb.shape[0]))
 return e, {'periods_px':periods}

def thin(a):
 z=np.pad(a.astype(bool),1)
 for _ in range(100):
  changed=False
  for phase in [0,1]:
   v=[z[:-2,1:-1],z[:-2,2:],z[1:-1,2:],z[2:,2:],z[2:,1:-1],z[2:,:-2],z[1:-1,:-2],z[:-2,:-2]]
   n=sum(x.astype(np.uint8) for x in v);b=sum((~v[i]&v[(i+1)%8]).astype(np.uint8) for i in range(8))
   if phase==0:c=(~(v[0]&v[2]&v[4]))&(~(v[2]&v[4]&v[6]))
   else:c=(~(v[0]&v[2]&v[6]))&(~(v[0]&v[4]&v[6]))
   remove=z[1:-1,1:-1]&(n>=2)&(n<=6)&(b==1)&c
   changed|=remove.any();z[1:-1,1:-1][remove]=False
  if not changed:break
 return z[1:-1,1:-1]

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
   if len(route)>=6 and np.ptp(xx)<=2 and np.ptp(yy)>2*np.ptp(xx)+4 and not events[xx].any():
    remove[yy[:-2],xx[:-2]]=True
  retained=sk&~remove
  distkeep=cv2.distanceTransform((~retained).astype(np.uint8),cv2.DIST_L2,3)
  distdrop=cv2.distanceTransform((~remove).astype(np.uint8),cv2.DIST_L2,3)
  cut=m&(distdrop<distkeep)&(distdrop<3)
  gp.summarize(t,m&~cut,t['ambiguous']|cut.any(0))

def extract(context):
 raw,strength,metrics=previous.evidence(context.image)
 leads=st.locate_leads(raw,context.config,context.geometry)
 original=[st.recover_stroke_pixels(raw>0,st.trace_lead(raw,strength,l),l) for l in leads]
 e,report=evidence(context.image);a=(e>16).astype(np.uint8)
 ts=[];changed=[]
 for l,t in zip(leads,original):
  spans=np.nan_to_num(t['y_max_px']-t['y_min_px']);tall=spans>l['row_spacing_px']*.5
  groups=int(np.sum(tall&~np.r_[False,tall[:-1]]))
  if groups>=3 and tall.sum()>=6:
   t=st.recover_stroke_pixels(a>0,trace_lead(a,e,l),l);changed.append(l['name'])
  ts.append(t)
 prune(ts,leads)
 report['refined']=changed;report['ownership']=own.separate_traces(ts,leads,context.image.shape)
 return Extraction(leads,ts,report)
