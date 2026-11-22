"""Experimental vertical periodic background for small horizontal waveforms."""
import numpy as np
import cv2
from scipy.signal import find_peaks
from .contracts import MethodAbstention

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
 bg=backgrounds[1];res=np.maximum(0,bg-align)
 e=cv2.warpAffine(res[:,:,0],cv2.invertAffineTransform(mat),(rgb.shape[1],rgb.shape[0]))
 return e, {'periods_px':periods}
