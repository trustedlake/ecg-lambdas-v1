"""Grid-color specialist: measured pixel paths for repeated tall strokes.
No source identity, lead-name rules, or production registration.
"""
from copy import deepcopy
import cv2
import numpy as np
from scipy.ndimage import percentile_filter, median_filter, maximum_filter1d
from . import faint_ink_ownership as ownership

def local_grid_evidence(rgb):
    f=rgb.astype(np.float32);chroma=f[:,:,0]-f[:,:,1]
    lines=cv2.HoughLinesP((chroma>18).astype(np.uint8),1,np.pi/1800,50,minLineLength=150,maxLineGap=10)
    hs=[];vs=[]
    for x0,y0,x1,y1 in ([] if lines is None else lines.reshape(-1,4)):
     dx=x1-x0;dy=y1-y0
     if abs(dx)>10*abs(dy):hs.append(dy/dx)
     if abs(dy)>10*abs(dx):vs.append(dx/dy)
    if not hs or not vs:
     return None
    mat=np.array([[1,-np.median(vs),0],[-np.median(hs),1,0]],np.float32)
    align=cv2.warpAffine(f,mat,(rgb.shape[1],rgb.shape[0]),borderValue=(255,255,255))
    density=-np.log(np.clip((align+1)/256,.01,1))
    chroma=(density[:,:,1]-density[:,:,0])*255;dark=density[:,:,0]*255
    vm=percentile_filter(dark,50,size=(151,1));hm=percentile_filter(dark,50,size=(1,81))
    vc=percentile_filter(chroma,50,size=(151,1));hc=percentile_filter(chroma,50,size=(1,81))
    ratio=np.maximum(np.where(vc>10,vm/np.maximum(vc,1),0),np.where(hc>10,hm/np.maximum(hc,1),0))
    grid=ratio*np.maximum(0,chroma)
    e=cv2.warpAffine(np.maximum(0,dark-grid),cv2.invertAffineTransform(mat),(rgb.shape[1],rgb.shape[0]))
    return e

def path(cost,vertical,base,step):
 h,w=cost.shape;rows=np.arange(h);prev=10*np.maximum(0,abs(rows-base)/step-.25);parents=np.zeros((h,w),np.int16)
 for x in range(w):
  initial=prev+cost[:,x]
  acc=np.cumsum(vertical[:,x]);v=initial-acc
  mn=np.minimum.accumulate(v);ii=np.maximum.accumulate(np.where(v<=mn,rows,0));down=acc+mn
  rev=np.cumsum(vertical[::-1,x]);v=initial[::-1]-rev
  mn=np.minimum.accumulate(v);jj=(h-1-np.maximum.accumulate(np.where(v<=mn,rows,0)))[::-1];up=(rev+mn)[::-1]
  choose=down<=up;parents[:,x]=np.where(choose,ii,jj);prev=np.minimum(down,up)
 end=int(np.argmin(prev+10*np.maximum(0,abs(rows-base)/step-.25)))
 result=np.zeros((h,w),np.uint8)
 for x in range(w-1,-1,-1):
  enter=int(parents[end,x]);result[min(end,enter):max(end,enter)+1,x]=1;end=enter
 return result


def summarize(t,mask,ambiguous=None):
 x,y=t['mask_origin'];h,w=mask.shape
 observed=mask.any(0);center=np.full(w,np.nan,np.float32);low=center.copy();high=center.copy()
 for col in np.flatnonzero(observed):
  points=np.flatnonzero(mask[:,col])+y
  center[col]=np.median(points);low[col]=points[0];high[col]=points[-1]
 if ambiguous is None:ambiguous=t['ambiguous'].copy()
 t.update(mask=mask,observed=observed,y_px=center,y_min_px=low,y_max_px=high,
          ambiguous=ambiguous,valid=observed&~ambiguous,coverage=float(observed.mean()),
          unambiguous_fraction=float((observed&~ambiguous).mean()))


def refine(rgb,leads,traces):
 e=local_grid_evidence(rgb)
 result=deepcopy(traces);changed=[]
 if e is None:return result,{'refined':changed,'abstained':'No two-axis colored grid'}
 for l,t in zip(leads,result):
  step=l['row_spacing_px'];base=l['baseline_y_px'];spans=np.nan_to_num(t['y_max_px']-t['y_min_px'])
  tall=spans>step*.5
  groups=int(np.sum(tall&~np.r_[False,tall[:-1]]))
  if groups<3 or tall.sum()<max(6,len(tall)*.01):continue
  x0,y0=t['mask_origin'];h,w=t['mask'].shape;x1=x0+w
  search_h=min(h,max(1,int(base+.55*step)-y0))
  yy=np.arange(y0,y0+search_h);local=e[y0:y0+search_h,x0:x1]
  reference=t['y_px'].copy();ok=np.isfinite(reference)
  if not ok.any():continue
  reference=np.interp(np.arange(w),np.flatnonzero(ok),reference[ok]);reference=median_filter(reference,size=3)
  events=maximum_filter1d((spans>step*.3).astype(np.uint8),size=max(5,int(step*.2)|1))
  cost=np.exp(-local/8)+.025
  horizontal=cost+.15*abs(yy[:,None]-reference[None,:])
  route=path(horizontal,(cost-.7*(local>15)*events[None,:])*.8,base-y0,step)
  selected=(cv2.dilate(route,np.ones((3,3),np.uint8))>0)&(local>3)
  mask=np.zeros((h,w),bool);mask[:search_h]=selected
  summarize(t,mask)
  t['method']='measured_grid_color_pixel_path'
  changed.append(l['name'])
 return result,{'refined':changed,'method':'local_grid_ratio_and_tall_stroke_paths'}
