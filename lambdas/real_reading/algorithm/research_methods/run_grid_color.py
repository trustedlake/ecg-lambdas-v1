"""Run the experimental extractor locally with an explicit layout configuration."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from .contracts import Context
from .chromatic_residual import extract
from .frozen_peaks34.ecg.preprocess import prepare
from .frozen_peaks34.ecg.grid import estimate_grid

def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text(encoding='utf8'))
    rgb,geometry=prepare(Path(a.image),config)
    result=extract(Context(rgb,config,geometry,estimate_grid(rgb,config)))
    integrity=result.validate(rgb.shape)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    mask=np.zeros(rgb.shape[:2],bool)
    for trace in result.traces:
        x,y=trace['mask_origin'];h,w=trace['mask'].shape;mask[y:y+h,x:x+w]|=trace['mask']
    paint=rgb.copy();paint[mask]=[0,100,255];Image.fromarray(paint).save(out/'extraction.png')
    np.savez_compressed(out/'mask.npz',mask=mask)
    (out/'report.json').write_text(json.dumps({'status':'experimental; not visually accepted','integrity':integrity,'diagnostics':result.diagnostics},indent=2))
    print(json.dumps(integrity))
if __name__=='__main__':main()
