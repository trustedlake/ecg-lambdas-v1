"""Resolve quarter turns from explicit page orientation and row geometry."""
import cv2
import numpy as np

def orient_landscape(a,config,method):
    h,w=a.shape[:2];identity=np.eye(3)
    info={'applied_clockwise_degrees':0,'uncertain':False}
    if config.get('expected_orientation')!='landscape' or h<=w*1.15:return a,identity,info
    priors=config['baseline_fractions'] if method=='colored_grid_boundary' else config.get('paper_baseline_fractions',config['baseline_fractions'])
    candidates=[]
    for degrees,turns,matrix in [(90,-1,[[0,-1,h-1],[1,0,0],[0,0,1]]),(270,1,[[0,1,0],[-1,0,w-1],[0,0,1]])]:
        rotated=np.ascontiguousarray(np.rot90(a,turns));rh,rw=rotated.shape[:2]
        f=rotated.astype(np.float32);gray=np.clip(f.max(2)+1.4*(f.max(2)-f.min(2)),0,255)
        contrast=cv2.GaussianBlur(gray,(0,0),7)-gray
        profile=(contrast[:,int(.10*rw):int(.94*rw)]>8).mean(1)
        score=sum(float(np.max(profile[max(0,int((p-.035)*rh)):min(rh,int((p+.035)*rh)+1)])) for p in priors)
        candidates.append((score,degrees,rotated,np.asarray(matrix,dtype=float)))
    candidates.sort(key=lambda item:item[0],reverse=True);best,second=candidates
    info['candidate_scores']={str(c[1]):c[0] for c in candidates}
    if best[0]<.05 or best[0]<second[0]*1.10:
        info['uncertain']=True;return a,identity,info
    info['applied_clockwise_degrees']=best[1]
    return best[2],best[3],info
