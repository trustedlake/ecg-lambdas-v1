"""Grid-color specialist: measured pixel paths for repeated tall strokes.
No source identity, lead-name rules, or production registration.
"""
from copy import deepcopy
import cv2
import numpy as np
from scipy.ndimage import percentile_filter, median_filter, maximum_filter1d
from . import faint_ink_ownership as ownership

def local_grid_evidence(rgb):
    f=rgb.astype(np.float32);d=255-f
    red=(f[:,:,0]-f[:,:,1]>25)&(f[:,:,0]>150)&(f[:,:,0]<245)
    if red.sum()<100:return None
    # Fit the reddish grid direction jointly in RGB, then remove that component.
    samples=d[red];samples=samples/np.maximum(samples[:,1:2],1)
    v=np.median(samples,axis=0);ink=np.ones(3,dtype=np.float32)
    design=np.stack([v,ink],axis=1);weights=np.linalg.pinv(design)[1]
    return np.maximum(0,d@weights)

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
