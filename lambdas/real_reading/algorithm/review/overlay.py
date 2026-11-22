"""One final presentation: rectified photo, trace labels and calibration grid."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def font(size):
    for p in ['C:/Windows/Fonts/segoeui.ttf','C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf']:
        try:return ImageFont.truetype(p,size)
        except OSError:pass
    return ImageFont.load_default()


def overlay_geometry(w,h):
    scale=w/1800.;pad=max(14,int(22*scale));header=max(86,int(106*scale));footer=max(58,int(70*scale))
    return [pad,header],[w+2*pad,h+header+footer]

def render(a,leads,traces,grid,config,status):
    h,w=a.shape[:2];scale=w/1800.;pad=max(14,int(22*scale));header=max(86,int(106*scale));footer=max(58,int(70*scale))
    canvas=Image.new('RGB',(w+2*pad,h+header+footer),'#eef3f7')
    d=ImageDraw.Draw(canvas)
    d.text((pad,12*scale),'ECG | Leads and calibration',font=font(max(22,int(30*scale))),fill='#17344c')
    cal=config['calibration'];label=f"{config['layout_label']}  |  {cal['speed_mm_s']} mm/s  |  {cal['gain_mm_mv']} mm/mV"
    d.text((pad,max(48,52*scale)),label,font=font(max(14,int(19*scale))),fill='#425d73')
    base=Image.fromarray(a).convert('RGBA');layer=Image.new('RGBA',base.size);g=ImageDraw.Draw(layer)
    if grid['valid']:
        for axis,length in [('x',w),('y',h)]:
            item=grid[axis];spacing=item['spacing_px']*5;phase=item['phase_px']
            for p in np.arange(phase,length,spacing):
                xy=(p,0,p,h) if axis=='x' else (0,p,w,p)
                g.line(xy,fill=(26,118,178,34),width=1)
        # Four visual calibration references, no editor in this version.
        cx=int(w*.48);cy=int(h*.10)
        xphase=grid['x']['phase_px'];yphase=grid['y']['phase_px']
        xs=grid['x']['spacing_px'];ys=grid['y']['spacing_px']
        cx=xphase+round((cx-xphase)/(5*xs))*5*xs;cy=yphase+round((cy-yphase)/(5*ys))*5*ys
        for x in [cx,cx+5*xs]:g.line((x,cy-ys,x,cy+6*ys),fill=(20,101,167,220),width=max(1,int(scale*2)))
        for y in [cy,cy+5*ys]:g.line((cx-xs,y,cx+6*xs,y),fill=(20,101,167,220),width=max(1,int(scale*2)))
    base=Image.alpha_composite(base,layer).convert('RGB');pixels=np.array(base)
    for lead,trace in zip(leads,traces):
        x,y=trace['mask_origin'];mh,mw=trace['mask'].shape;color=tuple(int(lead['color'][i:i+2],16) for i in (1,3,5))
        pixels[y:y+mh,x:x+mw][trace['mask']]=color
    base=Image.fromarray(pixels);d=ImageDraw.Draw(base);fs=max(22,int(38*scale))
    for lead,trace in zip(leads,traces):
        x0,y0,x1,y1=lead['box'];color=lead['color'];name=lead['name'].replace('_longo',' rhythm')
        d.rectangle((x0,y0,x1-1,y1-1),outline=color,width=max(3,int(3*scale)))
        box=d.textbbox((0,0),name,font=font(fs));tw=box[2]+14;th=fs+12
        d.rounded_rectangle((x0+3,y0+3,x0+tw,y0+th),radius=4,fill='white',outline=color)
        d.text((x0+9,y0+6),name,font=font(fs),fill=color)
    if grid['valid']:  # ONE large square (5 mm x 5 mm) so the grid can be checked by eye
        t=max(3,int(3*scale));purple=(190,40,255)
        gx0=grid['x']['phase_px']+int((w/2-grid['x']['phase_px'])//(5*grid['x']['spacing_px']))*5*grid['x']['spacing_px']
        gy0=grid['y']['phase_px']+int((h/2-grid['y']['phase_px'])//(5*grid['y']['spacing_px']))*5*grid['y']['spacing_px']
        d.rectangle((gx0,gy0,gx0+5*grid['x']['spacing_px'],gy0+5*grid['y']['spacing_px']),outline=purple,width=t)
        text='Purple box = exactly 1 square? If not, the grid is wrong -> mark No.';tf=font(max(16,int(26*scale)));tb=d.textbbox((0,0),text,font=tf)
        ty=h-tb[3]-16;d.rectangle((5,ty-6,tb[2]+19,ty+tb[3]+8),fill='white');d.text((12,ty),text,font=tf,fill=purple)
        ex,ey=gx0+2.5*grid['x']['spacing_px'],gy0+5*grid['y']['spacing_px']
        sx,sy=tb[2]/2,ty-6;n=max(2,int(((ex-sx)**2+(ey-sy)**2)**.5//(8*scale+6)))
        for i in range(0,n,2):d.line((sx+(ex-sx)*i/n,sy+(ey-sy)*i/n,sx+(ex-sx)*(i+1)/n,sy+(ey-sy)*(i+1)/n),fill=purple,width=t)
    canvas.paste(base,(pad,header));d=ImageDraw.Draw(canvas)
    label='Grid: consistent estimate' if grid['valid'] else 'Grid: calibration pending - physical arrays unavailable'
    d.text((pad,header+h+8),label,font=font(max(12,int(17*scale))),fill='#28546c' if grid['valid'] else '#a35b1a')
    d.text((pad,header+h+max(34,32*scale)),'Trace approximated by contrast | Gaps preserved | '+status,font=font(max(11,int(15*scale))),fill='#536879')
    return canvas
