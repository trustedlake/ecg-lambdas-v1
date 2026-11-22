"""Estimate grid geometry only. Speed and gain are never inferred here."""
import cv2
import numpy as np
from .preprocess import red_evidence


def periodicity(profile,expected_major_px=None):
    p=np.asarray(profile,np.float64)
    p-=cv2.GaussianBlur(p.reshape(1,-1),(0,0),12).ravel()
    if np.std(p)<.04 or len(p)<80: return None
    spectrum=np.fft.rfft(p,n=2*len(p));ac=np.fft.irfft(spectrum*np.conj(spectrum))[:len(p)]
    ac/=np.maximum(1,np.arange(len(p),0,-1));ac/=max(ac[0],1e-12)
    # Detect the stronger major grid. Its physical interval is explicit in config.
    hi=min(110,len(p)//3)
    low=12
    if expected_major_px:
        low=max(3,int(expected_major_px*.72));hi=min(hi,int(expected_major_px*1.30)+1)
    peaks=[k for k in range(low,hi) if ac[k]>=ac[k-1] and ac[k]>ac[k+1]]
    if not peaks:return None
    best=max(ac[k] for k in peaks)
    candidates=[k for k in peaks if ac[k]>max(.30,best*.78)]
    if not candidates:return None
    k=min(candidates)
    den=ac[k-1]-2*ac[k]+ac[k+1]
    frac=.5*(ac[k-1]-ac[k+1])/den if abs(den)>1e-9 else 0
    period=float(k+np.clip(frac,-.5,.5))
    # Refine from several repetitions to reduce integer-pixel quantization.
    ns=[];lags=[]
    for n in range(1,6):
        predicted=int(round(n*period))
        if predicted+3>=len(p)//2:break
        a=max(2,predicted-2);b=min(len(ac)-2,predicted+3)
        j=a+int(np.argmax(ac[a:b]))
        den=ac[j-1]-2*ac[j]+ac[j+1]
        offset=.5*(ac[j-1]-ac[j+1])/den if abs(den)>1e-9 else 0
        ns.append(n);lags.append(j+np.clip(offset,-.5,.5))
    if ns:period=float(np.dot(ns,lags)/np.dot(ns,ns))
    # Validate repetition at the second period where available.
    k2=int(round(2*period));score=float(min(ac[k],max(ac[max(1,k2-1):k2+2])))
    return period,score


def estimate_grid(a,config):
    h,w=a.shape[:2];f=a.astype(np.float32)
    chroma=np.maximum(0,np.minimum(f[:,:,0]-f[:,:,1],f[:,:,0]-f[:,:,2]))
    if np.mean(red_evidence(a))>.002:
        evidence=chroma; method='colored_grid_local_autocorrelation'
    else:
        gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32)
        evidence=cv2.GaussianBlur(gray,(0,0),5)-gray; method='monochrome_local_autocorrelation'
    observations={'x':[],'y':[]}
    for iy in range(3):
        for ix in range(3):
            x0=int(w*(.05+ix*.3));x1=int(w*(.32+ix*.3))
            y0=int(h*(.05+iy*.3));y1=int(h*(.32+iy*.3))
            tile=evidence[y0:y1,x0:x1]
            for axis,profile in [('x',tile.mean(0)),('y',tile.mean(1))]:
                expected=w/config['grid_width_mm_hint']*config['minor_square_mm']*config['major_interval_minor_squares']
                result=periodicity(profile,expected)
                if result:observations[axis].append({'tile':[ix,iy],'spacing_px':result[0]/config['major_interval_minor_squares'],'score':result[1]})
    out={'method':method,'harmonic_disambiguation_width_mm_hint':config['grid_width_mm_hint'],'measured_interval':'major_grid','major_interval_minor_squares':config['major_interval_minor_squares'],'minor_square_mm':config.get('minor_square_mm',1.),'observations':observations}
    ok=[]
    for axis in ['x','y']:
        obs=observations[axis];values=np.array([v['spacing_px'] for v in obs]);scores=np.array([v['score'] for v in obs])
        if len(values):
            med=float(np.median(values));keep=abs(values-med)/med<.15
            spacing=float(np.median(values[keep]));cv=float(np.std(values[keep])/spacing)
            agreement=float(np.mean(keep));score=float(np.median(scores[keep]))
            # Phase fitted to local line evidence. Controls define five minor cells.
            profile=evidence.mean(0 if axis=='x' else 1)
            major=5*spacing;phases=np.arange(0,major,.25)
            phase=float(max(phases,key=lambda p:np.mean(np.interp(np.arange(p,len(profile),major),np.arange(len(profile)),profile))))
            good=len(values)>=4 and agreement>=.67 and cv<.08 and score>.24
            out[axis]={'spacing_px':spacing,'pixels_per_mm':spacing/out['minor_square_mm'],'phase_px':phase,
                       'tile_agreement':agreement,'relative_dispersion':cv,'periodicity_score':score,'valid':good}
        else:
            out[axis]={'spacing_px':None,'pixels_per_mm':None,'phase_px':None,'valid':False}
        ok.append(out[axis]['valid'])
    ratio=(out['x']['spacing_px']/out['y']['spacing_px']) if all(out[a]['spacing_px'] for a in ['x','y']) else None
    # Planar photos can have differing x/y scale; reject only gross aliasing.
    out['valid']=bool(all(ok) and ratio is not None and .65<ratio<1.55)
    out['status']='estimated_consistent' if out['valid'] else 'review_required'
    out['controls']={a:[out[a]['phase_px'],out[a]['phase_px']+5*out[a]['spacing_px']] if out[a]['spacing_px'] else None for a in ['x','y']}
    out['control_interval_minor_squares']=5
    out['limitations']=['Periodicity is a consistency check, not independently annotated ground truth.',
                       'A planar grid does not correct curved paper or local photographic distortion.']
    return out
