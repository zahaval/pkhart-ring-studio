"""Put the supplied photo's own background back. The stone-free ring (as committed) is laid over the original photo, so
the floor and its soft shadow are the photo's own pixels. Only where the photo's stones stuck out past the bands, or the
pear sat on the floor, is the floor filled in, smoothly, from the floor around it. No whitening, no other clean-up."""
import os, sys, numpy as np
from PIL import Image
from scipy import ndimage
R='/Users/zahavalowy/Projects/pkhart-ring-studio/'; D=os.path.dirname(os.path.abspath(__file__))
src=open(R+'tools/make_ring_photos.py').read(); ns={'__file__':R+'tools/make_ring_photos.py'}; exec(src[:src.index('def erase_curved')],ns); tint=ns['tint']
for n in range(1,6):
    std=np.asarray(Image.open(f'{D}/std-{n}.png').convert('RGB')).astype(float); cur=np.asarray(Image.open(f'{D}/cur/ring-{n}.jpg').convert('RGB')).astype(float)
    mx=cur.max(-1); mn=cur.min(-1); gold=((mx-mn)/np.maximum(mx,1)>0.2)&(cur[...,0]-cur[...,2]>35)
    lab,k=ndimage.label(gold); sz=ndimage.sum(gold,lab,range(1,k+1)); gold=np.isin(lab,1+np.nonzero(sz>400)[0])
    ring=ndimage.binary_dilation(ndimage.binary_closing(gold,iterations=2),iterations=3)
    d=np.abs(std-cur).max(-1); old=ndimage.binary_fill_holes(ndimage.binary_closing((d>28)&~ring,iterations=6)|ring)&~ring
    old=ndimage.binary_dilation(old,iterations=5)&~ring
    smx=std.max(-1); smn=std.min(-1); ssat=(smx-smn)/np.maximum(smx,1)          # the stones' own faint coloured or grey fringe just outside that
    fringe=((ssat>0.05)|(std.mean(-1)<228))&ndimage.binary_dilation(old,iterations=12)&~ring
    old=ndimage.binary_dilation(old|fringe,iterations=2)&~ring
    ok=(~old&~ring).astype(float); fill=np.zeros_like(std); got=np.zeros(std.shape[:2],bool)
    for sg in (6,14,30,60):                                    # the floor around each hole, carried in smoothly
        w=ndimage.gaussian_filter(ok,sg); v=np.stack([ndimage.gaussian_filter(std[...,c]*ok,sg) for c in range(3)],-1)/np.maximum(w,1e-6)[...,None]
        take=(w>0.08)&~got; fill[take]=v[take]; got|=take
    fill[~got]=255
    # then relax it so it meets the photo's floor exactly at the hole's edge (no lighter or darker patch)
    lab2,k2=ndimage.label(old); kern=np.array([[0,1,0],[1,0,1],[0,1,0]],float)
    for i,sl in enumerate(ndimage.find_objects(lab2)):
        y0,y1,x0,x1=max(0,sl[0].start-3),min(std.shape[0],sl[0].stop+3),max(0,sl[1].start-3),min(std.shape[1],sl[1].stop+3)
        hole=lab2[y0:y1,x0:x1]==i+1; use=(ok[y0:y1,x0:x1]>0)|hole; sub=np.where(hole[...,None],fill[y0:y1,x0:x1],std[y0:y1,x0:x1]); u=use.astype(float)
        den=np.maximum(ndimage.convolve(u,kern,mode='nearest'),1e-6)[...,None]
        for it in range(min(2500,60+4*max(hole.shape))):
            num=np.stack([ndimage.convolve(sub[...,c]*u,kern,mode='nearest') for c in range(3)],-1); sub[hole]=(num/den)[hole]
        fill[y0:y1,x0:x1][hole]=sub[hole]
    mo=ndimage.gaussian_filter(old.astype(float),1.0)[...,None]; mr=ndimage.gaussian_filter(ring.astype(float),0.8)[...,None]
    out=std*(1-mo)+fill*mo; out=out*(1-mr)+cur*mr
    # the photo's paper white (253-254) becomes the page's white, and the frame's edge fades out, so no box shows on the page
    mnn=out.min(-1); mxx=out.max(-1); t=np.clip((mnn-250)/4.0,0,1)*(mxx-mnn<6); out=out+(255-out)*t[...,None]
    H,W=out.shape[:2]; yy,xx=np.mgrid[:H,:W]; e=np.clip(np.minimum(np.minimum(xx,W-1-xx),np.minimum(yy,H-1-yy))/40.0,0,1); out=255-(255-out)*np.where(ring,1.0,e)[...,None]
    print(n,'ring px',int(ring.sum()),'filled floor px',int(old.sum()))
    for metal in ('gold','rose','silver'):
        px=out if metal=='gold' else tint(out,metal,None)
        Image.fromarray(np.clip(px,0,255).astype(np.uint8)).save(R+f'img/rings/{metal}/ring-{n}.jpg',quality=93)
sh=Image.new('RGB',(5*520,3*520),'white')
for n in range(1,6):
    for j,p in enumerate((f'{D}/std-{n}.png',R+f'img/rings/gold/ring-{n}.jpg')):
        a=np.asarray(Image.open(p).convert('RGB')).astype(float); sh.paste(Image.fromarray(np.clip(255-(255-a)*4,0,255).astype(np.uint8)).resize((520,520)),((n-1)*520,j*520))
    sh.paste(Image.open(R+f'img/rings/gold/ring-{n}.jpg').resize((520,520)),((n-1)*520,2*520))
sh.save(f'{D}/cmp2.jpg',quality=88)
