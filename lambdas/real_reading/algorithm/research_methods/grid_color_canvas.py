"""Specialist canvas retaining source pixels outside the detected grid boundary."""
from copy import deepcopy
import cv2
import numpy as np
from .frozen_peaks34.ecg.preprocess import prepare as frozen_prepare, decode

def prepare(source, config, page=0):
    original, geometry = frozen_prepare(source, config, page)
    cfg = deepcopy(config)
    h, w = original.shape[:2]
    if geometry['method'] != 'colored_grid_boundary':
        return original, geometry, cfg
    # One row of vertical context; no lead identity or image identity is used.
    fractions = np.asarray(config['baseline_fractions'])
    step = np.median(np.diff(fractions)) if len(fractions)>1 else .08
    my = max(8, int(np.ceil(h*step)))
    mx = max(8, int(np.ceil(w*.02)))
    nh, nw = h+2*my, w+2*mx
    pixels, source_size = decode(source, page, config.get('max_side',2400))
    scale = np.diag([source_size[0]/pixels.shape[1],source_size[1]/pixels.shape[0],1.])
    translate = np.array([[1.,0,mx],[0,1.,my],[0,0,1.]])
    transform = translate @ np.asarray(geometry['source_to_working'])
    rgb = cv2.warpPerspective(pixels, transform @ scale, (nw,nh), borderValue=(255,255,255))
    for key in ('baseline_fractions','paper_baseline_fractions'):
        if key in cfg: cfg[key] = [(v*h+my)/nh for v in cfg[key]]
    cfg['signal_x_fraction'] = [(v*w+mx)/nw for v in config.get('signal_x_fraction',[.04,.97])]
    cfg['label_exclusion_fraction'] = config.get('label_exclusion_fraction',.008)*w/nw
    geometry = deepcopy(geometry)
    geometry.update(working_size=[nw,nh],source_to_working=transform.tolist(),working_to_source=np.linalg.inv(transform).tolist(),retained_margin_px=[mx,my],grid_canvas_size=[w,h])
    return rgb, geometry, cfg
