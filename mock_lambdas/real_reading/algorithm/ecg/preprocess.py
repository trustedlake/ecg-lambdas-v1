"""Decode, locate grid/paper and rectify a plane. Never edits the source."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from .orientation import orient_landscape

FORMATS={'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp','.pdf'}

def page_count(source):
    if Path(source).suffix.lower()=='.pdf':
        import pymupdf
        with pymupdf.open(source) as doc: return len(doc)
    with Image.open(source) as im: return getattr(im,'n_frames',1)


def decode(source,page=0,max_side=2400):
    if Path(source).suffix.lower()=='.pdf':
        import pymupdf
        with pymupdf.open(source) as doc:
            p=doc[page]; scale=min(3,max_side/max(p.rect.width,p.rect.height))
            pix=p.get_pixmap(matrix=pymupdf.Matrix(scale,scale),colorspace=pymupdf.csRGB,alpha=False)
            im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    else:
        with Image.open(source) as original:
            original.seek(page)
            im=ImageOps.exif_transpose(original).convert('RGB')
    original_size=list(im.size)
    im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    return np.array(im),original_size


def order_quad(points):
    p=np.asarray(points,dtype=np.float32).reshape(4,2)
    # Angular order avoids duplicate corners for strongly rotated rectangles.
    center=p.mean(0);p=p[np.argsort(np.arctan2(p[:,1]-center[1],p[:,0]-center[0]))]
    p=np.roll(p,-np.argmin(p.sum(1)),axis=0)
    u=p[1]-p[0];v=p[2]-p[1]
    if u[0]*v[1]-u[1]*v[0]<0: p=p[[0,3,2,1]]
    return p


def red_evidence(a):
    f=a.astype(np.float32); r,g,b=f[:,:,0],f[:,:,1],f[:,:,2]
    chroma=np.maximum(0,np.minimum(r-g,r-b))
    # Remove broad paper/illumination coloration, retain colored grid strokes.
    residual=chroma-cv2.GaussianBlur(chroma,(0,0),7)
    return ((chroma>4)&(residual>1.3)).astype('uint8')


def locate_quad(a):
    h,w=a.shape[:2]; s=min(1,1200/max(h,w)); small=cv2.resize(a,None,fx=s,fy=s)
    hh,ww=small.shape[:2]; red=red_evidence(small)
    hs_scale=min(1,600/ww)
    line_mask=cv2.resize(red*255,None,fx=hs_scale,fy=hs_scale,interpolation=cv2.INTER_NEAREST)
    lines=cv2.HoughLinesP(line_mask,1,np.pi/1800,threshold=35,minLineLength=max(60,int(ww*hs_scale*.20)),maxLineGap=8)
    hs=[];vs=[]
    if lines is not None:
        for x0,y0,x1,y1 in lines.reshape(-1,4):
            dx=x1-x0;dy=y1-y0
            if abs(dx)>abs(dy)*8:hs.append(dy/dx)
            if abs(dy)>abs(dx)*8:vs.append(dx/dy)
    axis_aligned=bool(len(hs)>5 and len(vs)>5 and abs(np.median(hs))<.007 and abs(np.median(vs))<.007)
    mask=cv2.morphologyEx(red,cv2.MORPH_CLOSE,np.ones((13,13),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours,key=cv2.contourArea,reverse=True):
        if cv2.contourArea(cnt)<hh*ww*.16: break
        hull=cv2.convexHull(cnt); poly=cv2.approxPolyDP(hull,.018*cv2.arcLength(hull,True),True)
        if axis_aligned:
            x,y,bw,bh=cv2.boundingRect(cnt)
            return np.float32([[x,y],[x+bw-1,y],[x+bw-1,y+bh-1],[x,y+bh-1]])/s,'colored_grid_boundary'
        if len(poly)==4:
            return order_quad(poly.reshape(4,2)/s),'colored_grid_boundary'
    # For monochrome images, a bright paper quadrilateral is only a fallback.
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours,key=cv2.contourArea,reverse=True):
        area=cv2.contourArea(cnt)
        if not hh*ww*.22<area<hh*ww*.96: continue
        poly=cv2.approxPolyDP(cv2.convexHull(cnt),.02*cv2.arcLength(cv2.convexHull(cnt),True),True)
        if len(poly)==4: return order_quad(poly.reshape(4,2)/s),'paper_boundary_fallback'
    return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32),'full_image_fallback'


def prepare(source,config,page=0):
    a,original_size=decode(source,page,config.get('max_side',2400))
    h,w=a.shape[:2]
    if config.get('quad_normalized'):
        quad=order_quad(np.array(config['quad_normalized'])*[w-1,h-1]); method='configured_quad'
    else: quad,method=locate_quad(a)
    tl,tr,br,bl=quad
    nw=int(round(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))))+1
    nh=int(round(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))))+1
    if min(nw,nh)<100: raise ValueError('Detected image region too small')
    matrix=cv2.getPerspectiveTransform(quad,np.float32([[0,0],[nw-1,0],[nw-1,nh-1],[0,nh-1]]))
    rect=cv2.warpPerspective(a,matrix,(nw,nh),borderValue=(255,255,255))
    rect,rotation,orientation=orient_landscape(rect,config,method)
    nh,nw=rect.shape[:2]
    scale=np.diag([w/original_size[0],h/original_size[1],1.])
    transform=rotation@matrix@scale
    return rect,{'method':method,'source_coordinate_system':'decoded_EXIF_oriented_pixels','source_size':original_size,
                 'working_size':[nw,nh],'source_to_working':transform.tolist(),'working_to_source':np.linalg.inv(transform).tolist(),
                 'quad_source':(quad/[w/original_size[0],h/original_size[1]]).tolist(),
                 'orientation':orientation,'model':'planar_homography','nonlinear_paper_warp_corrected':False}
