"""Experimental vertical-background measured recovery for small waveforms."""
import numpy as np
import cv2
from . import grid_color_small_wave as previous, grid_color_small_wave_evidence as periodic, grid_color_paths as gp

def refine(rgb, result):
 eligible=result.diagnostics.get('small_wave_refined',[])
 if not eligible:return result
 evidence,_=periodic.evidence(rgb)
 changed=[]
 for lead,trace in zip(result.leads,result.traces):
  if lead['name'] not in eligible:continue
  step=lead['row_spacing_px']
  original_x0,x1=lead['trace_x'];base=lead['baseline_y_px']
  # Search the unused left margin of the first layout column for measured ink.
  x0=max(0,original_x0-round(step)) if lead['column']==0 else original_x0
  # Keep the extension inside the visibly colored grid, excluding page edges.
  if x0<original_x0:
   band=rgb[max(0,int(base-step*.4)):min(rgb.shape[0],int(base+step*.4))].astype(np.float32)
   chroma=(band[:,:,0]-band[:,:,1]).mean(0)
   chroma=cv2.GaussianBlur(chroma.reshape(1,-1),(5,1),0).ravel()
   support=np.flatnonzero(chroma[x0:original_x0]>3)
   x0=x0+int(support[0]) if support.size else original_x0
  extension=original_x0-x0
  y0=max(0,int(base-step*.45));y1=min(rgb.shape[0],int(base+step*.35))
  z=evidence[y0:y1,x0:x1];h,w=z.shape
  guide=trace['y_px'].copy();known=np.isfinite(guide)
  guide=np.interp(np.arange(x0,x1),trace['x_px'][known],guide[known])-y0
  # Interpolation constrains search only; exports still require measured evidence.
  cost=-np.minimum(z,30)/10+.8*np.maximum(0,abs(np.arange(h)[:,None]-guide[None,:])-2)
  if extension:cost[:,:extension]=-np.minimum(z[:,:extension],30)/10+.1*abs(np.arange(h)[:,None]-guide[None,:extension])
  prev=cost[:,0];back=np.zeros((h,w),np.int16);rr=np.arange(h)
  ds=np.arange(-6,7);idx=rr[:,None]+ds;valid=(idx>=0)&(idx<h)
  for x in range(1,w):
   v=prev[np.clip(idx,0,h-1)]+(.2 if x<extension else .35)*np.abs(ds);v[~valid]=1e6;j=v.argmin(1)
   back[:,x]=idx[rr,j];prev=cost[:,x]+v[rr,j]
  ys=np.zeros(w,int);ys[-1]=prev.argmin()
  for x in range(w-1,0,-1):ys[x-1]=back[ys[x],x]
  ox,oy=trace['mask_origin'];mask=np.zeros((trace['mask'].shape[0],w),bool);ox=x0
  trace['mask_origin']=[ox,oy];trace['x_px']=np.arange(x0,x1,dtype=np.float32)
  trace['ambiguous']=np.pad(trace['ambiguous'],(extension,0),constant_values=True)
  lead['trace_x']=[x0,x1]
  for x,y in enumerate(ys):
   near=ys[max(0,x-1):min(w,x+2)]
   for yy in range(max(0,min(near)-1),min(h,max(near)+2)):
    if z[yy,x]>3 and 0<=yy+y0-oy<mask.shape[0]:mask[yy+y0-oy,x+x0-ox]=True
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
 result.diagnostics['vertical_small_wave_refined']=changed
 return result

def extract(context):
 return refine(context.image,previous.extract(context))
