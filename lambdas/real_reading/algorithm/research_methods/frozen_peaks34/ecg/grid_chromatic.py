"""Measure individual colored grid lines; avoid aliasing four cells as five.

Only used by the explicit chromatic_ratio_paths profile. Invalid fits withhold
physical units via the existing conversion contract.
"""
import numpy as np
from scipy.signal import find_peaks


def fit_lines(profile, expected):
    p=np.asarray(profile,float)
    if len(p)<expected*12 or np.ptp(p)<2:return None
    peaks,_=find_peaks(p,prominence=max(2,np.ptp(p)*.18),distance=max(1,int(expected*.65)))
    if len(peaks)<12:return None
    gaps=np.diff(peaks); pitch=float(np.median(gaps))
    if not .75*expected<pitch<1.25*expected:return None
    indices=np.r_[0,np.cumsum(np.maximum(1,np.rint(gaps/pitch))).astype(int)]
    slope,phase=np.polyfit(indices,peaks,1)
    residual=peaks-(slope*indices+phase)
    rms=float(np.sqrt(np.mean(residual**2)))
    if rms>max(.6,slope*.12) or max(abs(residual))>slope*.30:return None
    return dict(spacing_px=float(slope),phase_px=float(phase%slope),rms_px=rms,lines=len(peaks))


def estimate_grid(a,config):
    f=a.astype(float);chroma=f.max(2)-f.min(2);h,w=chroma.shape
    expected=w/config['grid_width_mm_hint']*config['minor_square_mm']
    observations={'x':[],'y':[]}
    for iy in range(3):
        for ix in range(3):
            x0,x1=int(w*(.05+ix*.3)),int(w*(.32+ix*.3))
            y0,y1=int(h*(.05+iy*.3)),int(h*(.32+iy*.3))
            tile=chroma[y0:y1,x0:x1]
            for axis,profile,offset in [('x',tile.mean(0),x0),('y',tile.mean(1),y0)]:
                fit=fit_lines(profile,expected)
                if fit:
                    fit['phase_px']=(fit['phase_px']+offset)%fit['spacing_px']
                    observations[axis].append(dict(tile=[ix,iy],**fit))
    out=dict(method='chromatic_minor_line_regression',measured_interval='minor_grid',
        minor_square_mm=config['minor_square_mm'],major_interval_minor_squares=config['major_interval_minor_squares'],observations=observations)
    for axis,obs in observations.items():
        values=np.array([v['spacing_px'] for v in obs])
        if len(values):
            med=float(np.median(values));disp=float(np.std(values)/med)
            phase=float(obs[0]['phase_px']);valid=len(values)>=4 and disp<.03
            out[axis]=dict(spacing_px=med,pixels_per_mm=med/config['minor_square_mm'],phase_px=phase,
                relative_dispersion=disp,tile_agreement=len(values)/9,valid=valid)
        else:out[axis]=dict(spacing_px=None,pixels_per_mm=None,phase_px=None,valid=False)
    out['valid']=all(out[k]['valid'] for k in ['x','y'])
    out['status']='estimated_consistent' if out['valid'] else 'review_required'
    interval=config['major_interval_minor_squares']
    out['controls']={k:[out[k]['phase_px'],out[k]['phase_px']+interval*out[k]['spacing_px']] if out[k]['valid'] else None for k in ['x','y']}
    out['control_interval_minor_squares']=interval
    out['limitations']=['Line periodicity does not establish electrical accuracy or correct nonplanar paper distortion.']
    return out
