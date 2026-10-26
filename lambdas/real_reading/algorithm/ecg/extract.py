"""Pure pixel-to-physical conversion. No OCR and no inferred speed/gain."""
import numpy as np


def convert(trace,grid,calibration,selection=None,lead_name=None):
    if not calibration or not all(k in calibration for k in ('speed_mm_s','gain_mm_mv')):
        raise ValueError('Explicit speed_mm_s and gain_mm_mv are required (null means unavailable).')
    speed=calibration['speed_mm_s'];gain=calibration['gain_mm_mv']
    for value in (speed,gain):
        if value is not None and (not np.isfinite(value) or value<=0):raise ValueError('Calibration must be positive')
    x=trace['x_px'];y=trace['y_px'];valid=trace['valid'].copy()
    selection=selection or {};enabled=selection.get('enabled_leads')
    if enabled is not None and lead_name not in enabled:valid[:]=False
    for interval in selection.get('excluded_intervals',[]):
        if interval['lead']==lead_name:
            if interval.get('coordinate_system')!='working_x_px':raise ValueError('Exclusions require working_x_px coordinates')
            valid[(x>=interval['start'])&(x<=interval['end'])]=False
    out={k:trace[k] for k in ['x_px','y_px','y_min_px','y_max_px','observed','ambiguous']}
    out['selected_valid']=valid
    calibrated=bool(grid['valid'] and speed is not None and gain is not None)
    if calibrated:
        out['time_s']=(x-x[0])/(grid['x']['pixels_per_mm']*speed)
        out['amplitude_mv']=(trace['baseline_y_px']-y)/(grid['y']['pixels_per_mm']*gain)
        out['amplitude_min_mv']=(trace['baseline_y_px']-trace['y_max_px'])/(grid['y']['pixels_per_mm']*gain)
        out['amplitude_max_mv']=(trace['baseline_y_px']-trace['y_min_px'])/(grid['y']['pixels_per_mm']*gain)
        out['valid']=valid
    else:
        for key in ['time_s','amplitude_mv','amplitude_min_mv','amplitude_max_mv']:out[key]=np.full(len(x),np.nan,np.float32)
        out['valid']=np.zeros(len(x),bool)
    out['analysis_mv']=np.where(out['valid'],out['amplitude_mv'],np.nan)
    return out,calibrated
