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
    lb,kb=ndimage.label(old); szb=ndimage.sum(old,lb,range(1,kb+1)); big=np.isin(lb,1+np.nonzero(szb>6000)[0])
    old=(old|ndimage.binary_dilation(big,iterations=12))&~ring          # around the pear, out past its own dark edge to clean floor
    ok=(~old&~ring).astype(float); fill=np.zeros_like(std); got=np.zeros(std.shape[:2],bool)
    for sg in (6,14,30,60):                                    # the floor around each hole, carried in smoothly
        w=ndimage.gaussian_filter(ok,sg); v=np.stack([ndimage.gaussian_filter(std[...,c]*ok,sg) for c in range(3)],-1)/np.maximum(w,1e-6)[...,None]
        take=(w>0.08)&~got; fill[take]=v[take]; got|=take
    fill[~got]=255
    # each hole is solved exactly (Laplace): it takes the photo's floor at its edge and stays smooth inside, so there is
    # no lighter or darker patch where a stone or the pear was
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    lab2,k2=ndimage.label(old)
    for i,sl in enumerate(ndimage.find_objects(lab2)):
        y0,y1,x0,x1=max(0,sl[0].start-2),min(std.shape[0],sl[0].stop+2),max(0,sl[1].start-2),min(std.shape[1],sl[1].stop+2)
        hole=lab2[y0:y1,x0:x1]==i+1; known=(ok[y0:y1,x0:x1]>0)&~hole; sub=std[y0:y1,x0:x1]; hh,ww=hole.shape
        idx=-np.ones((hh,ww),int); ys,xs=np.nonzero(hole); idx[ys,xs]=np.arange(len(ys)); N=len(ys)
        rows=[];cols=[];vals=[]; diag=np.zeros(N); rhs=np.zeros((N,3))
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            yy2=ys+dy; xx2=xs+dx; inb=(yy2>=0)&(yy2<hh)&(xx2>=0)&(xx2<ww); p=np.nonzero(inb)[0]
            nb=idx[yy2[p],xx2[p]]; isu=nb>=0; kn=known[yy2[p],xx2[p]]
            rows+=list(p[isu]); cols+=list(nb[isu]); vals+=[-1.0]*int(isu.sum()); np.add.at(diag,p[isu],1.0)
            pk=p[kn&~isu]; np.add.at(diag,pk,1.0); np.add.at(rhs,pk,sub[yy2[pk],xx2[pk]])
        prior=fill[y0:y1,x0:x1][ys,xs]; diag=diag+1e-4; rhs=rhs+1e-4*prior          # a hole with no floor at its edge keeps the carried-in value
        A=csr_matrix((vals+list(diag),(rows+list(range(N)),cols+list(range(N)))),shape=(N,N))
        sol=np.stack([spsolve(A,rhs[:,c]) for c in range(3)],-1); fill[y0:y1,x0:x1][hole]=sol
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
