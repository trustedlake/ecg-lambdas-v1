"""Experimental measured-pixel paths for low-amplitude traces on periodic grids."""
import numpy as np
import cv2
from . import grid_color_continuity_guard as previous, grid_color_periodic as periodic, grid_color_paths as gp

def refine(rgb, result):
 # The narrow horizontal search is unsuitable when any small trace drifts
 # by more than a quarter of its row spacing across the page.
 slopes=[]
 for lead,trace in zip(result.leads,result.traces):
  obs=trace['observed'];step=lead['row_spacing_px']
  if obs.sum()<20: continue
  if np.nanpercentile(abs(trace['y_px']-trace['baseline_y_px']),95)<step*.25 and np.nanpercentile(trace['y_max_px']-trace['y_min_px'],99)<step*.3:
   slope=np.polyfit(trace['x_px'][obs],trace['y_px'][obs],1)[0]
   slopes.append(abs(slope)*len(obs)/step)
 if not slopes or max(slopes)>.25:
  result.diagnostics['small_wave_refined']=[]
  result.diagnostics['small_wave_abstention']='Row tilt exceeds horizontal search tolerance'
  return result
 evidence,_=periodic.evidence(rgb)
 changed=[]
 for lead,trace in zip(result.leads,result.traces):
  step=lead['row_spacing_px']; observed=trace['observed']
  if observed.sum()<20: continue
  amplitude=np.nanpercentile(abs(trace['y_px']-trace['baseline_y_px']),95)/step
  span=np.nanpercentile(trace['y_max_px']-trace['y_min_px'],99)/step
  # Fixed six-pixel path steps must not replace larger sampled peaks.
  if amplitude>=.25 or span>=.3 or np.nanpercentile(trace['y_max_px']-trace['y_min_px'],99)>12: continue
  x0,x1=lead['trace_x'];base=lead['baseline_y_px']
  y0=max(0,int(base-step*.45));y1=min(rgb.shape[0],int(base+step*.35))
  z=evidence[y0:y1,x0:x1];h,w=z.shape
  cost=-np.minimum(z,30)/10;prev=cost[:,0];back=np.zeros((h,w),np.int16);rr=np.arange(h)
  ds=np.arange(-6,7);idx=rr[:,None]+ds;valid=(idx>=0)&(idx<h)
  for x in range(1,w):
   v=prev[np.clip(idx,0,h-1)]+.5*np.abs(ds);v[~valid]=1e6;j=v.argmin(1)
   back[:,x]=idx[rr,j];prev=cost[:,x]+v[rr,j]
  ys=np.zeros(w,int);ys[-1]=prev.argmin()
  for x in range(w-1,0,-1):ys[x-1]=back[ys[x],x]
  mask=np.zeros_like(trace['mask']);ox,oy=trace['mask_origin']
  for x,y in enumerate(ys):
   for yy in range(max(0,y-1),min(h,y+2)):
    if z[yy,x]>8 and 0<=yy+y0-oy<mask.shape[0]:mask[yy+y0-oy,x+x0-ox]=True
  # Reject short, flat islands that depart from both neighboring measured
  # portions. Do not bridge the gap or draw a replacement segment.
  n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
  centers=np.full(mask.shape[1],np.nan)
  for col in np.flatnonzero(mask.any(0)):centers[col]=np.median(np.flatnonzero(mask[:,col]))
  for k in range(1,n):
   xx,yy,ww,hh,area=stats[k]
   if ww<8 or ww>step*.65 or hh>3:continue
   left=np.flatnonzero(np.isfinite(centers[max(0,xx-15):xx]))+max(0,xx-15)
   right=np.flatnonzero(np.isfinite(centers[xx+ww:min(len(centers),xx+ww+15)]))+xx+ww
   if left.size and right.size:
    a,b=left[-1],right[0];pred=centers[a]+(centers[b]-centers[a])*(xx+ww/2-a)/(b-a)
    if abs(yy+(hh-1)/2-pred)>2.5:mask[labels==k]=False
  # Path coordinates are internal. Only measured residual pixels are exported;
  # unsupported columns stay NaN and uncertain.
  gp.summarize(trace,mask,trace['ambiguous']|~mask.any(0))
  changed.append(lead['name'])
 result.diagnostics['small_wave_refined']=changed
 return result

def extract(context):
 return refine(context.image,previous.extract(context))
