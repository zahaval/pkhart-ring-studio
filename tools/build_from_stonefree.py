"""Ring photos from the stone-free renders (Artlist, Nano Banana 2: the supplied photo with only the stones removed).
Each render is put in the same 900px frame as the supplied photo it came from (same crop and scale), so the stone
positions in index.html do not change. Nothing else is edited: only the frame's outer edge fades to the page's white.
usage: build.py OUT_DIR   (writes OUT_DIR/<metal>/ring-N.jpg)"""
import json, os, sys, numpy as np
from PIL import Image
D=os.path.dirname(os.path.abspath(__file__)); R='/Users/zahavalowy/Projects/pkhart-ring-studio/'; OUT=sys.argv[1]
src=open(R+'tools/make_ring_photos.py').read(); ns={'__file__':R+'tools/make_ring_photos.py'}; exec(src[:src.index('def erase_curved')],ns); tint=ns['tint']
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
PEAR={1:[351,454,238,179],2:[344,548,228,177],3:[330,535,237,177],4:[334,554,232,173],5:[333,597,227,183]}      # where the photo's pear sat: x, y, w, h
SEAT={1:(589,509),2:(572,602),3:(567,574),4:(566,619),5:(560,670)}                                            # its right end, on the wire
def shorten(a,n):
    """The render lets the loose bottom wire run on under where the pear was. It is cut back to end just inside the
    pear's right end (as the real ring's wire does), with a rounded tip made from the wire's own cross-section; the
    floor where the cut-off piece lay is filled from the floor around it."""
    H,W=a.shape[:2]; mx=a.max(-1); mn=a.min(-1); g=((mx-mn)/np.maximum(mx,1)>0.25)&(a[...,0]-a[...,2]>45)
    px,py,pw,ph=PEAR[n]; sx,sy=SEAT[n]; x_end=px+pw-int(0.19*pw)
    def runs(x):
        ys=np.nonzero(g[:,x])[0]; out=[]
        if len(ys)==0: return out
        s0=p=ys[0]
        for y in ys[1:]:
            if y>p+2: out.append((s0,p)); s0=y
            p=y
        out.append((s0,p)); return [q for q in out if q[1]-q[0]>=5]
    r0=min(runs(sx+12),key=lambda q:abs((q[0]+q[1])/2.0-sy)); top,bot=r0; th=bot-top+1; T={}; x=sx+12
    while x>px-30:
        lo_,hi_=max(0,top-5),min(H,bot+9); ys=np.nonzero(g[lo_:hi_,x])[0]          # the wire here: first to last gold pixel near where it was (highlights included)
        if len(ys)<4: break
        t,b=lo_+ys.min(),lo_+ys.max()
        if b-t+1>th+6: c=(top+bot)/2.0; t=max(t,int(c-th/2.0-3)); b=min(b,int(c+th/2.0+3))
        if x<x_end and b-t+1<0.45*th: break                        # the wire's own tip
        T[x]=(t,b); top,bot=t,b; x-=1
    tip=min(T); half=th/2.0; print('  wire',n,'tip at',tip,'cut to',x_end,'thickness',th)
    if tip>=x_end-6: return a
    xc=x_end+int(half)+1; hole=np.zeros((H,W),bool); yy,xx=np.mgrid[:H,:W]
    for x in range(tip,xc):
        t,b=T[x]; hole[max(0,t-1):b+16,x]=True                         # the piece and the contact shadow under it
    topline=np.array([T[min(max(x,tip),xc)][0] for x in range(W)])     # anything else of the wire's end left below that line (its curl)
    rest=((mn<232)|((mx-mn)/np.maximum(mx,1)>0.08))&(xx>=px-70)&(xx<xc)&(yy>=topline[None,:]-5)&(yy<=py+ph+45)
    midline=np.array([(T[min(max(x,tip),xc)][0]+T[min(max(x,tip),xc)][1])/2.0 for x in range(W)])
    if n!=1: rest&=(xx>=tip-40)&(yy>=midline[None,:])&(yy<=midline[None,:]+60)      # elsewhere only what is left of the wire below its own centre line (bands are all above it)
    up=np.roll(hole,-2,0)&~hole; hole=ndimage.binary_dilation(hole|rest,iterations=3)&~ndimage.binary_dilation(up&g,iterations=3) if n!=1 else ndimage.binary_dilation(hole|rest,iterations=3); hole[:,xc:]=False
    known=~hole&(mn>222)&((mx-mn)<16)                 # plain floor only
    ys,xs=np.nonzero(hole); idx=-np.ones((H,W),int); idx[ys,xs]=np.arange(len(ys)); N=len(ys); rows=[];cols=[];vals=[]; diag=np.full(N,1e-4); rhs=np.full((N,3),1e-4*254.0)
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        y2=np.clip(ys+dy,0,H-1); x2=np.clip(xs+dx,0,W-1); nb=idx[y2,x2]; isu=nb>=0; kn=known[y2,x2]&~isu; p=np.arange(N)
        rows+=list(p[isu]); cols+=list(nb[isu]); vals+=[-1.0]*int(isu.sum()); np.add.at(diag,p[isu],1.0); np.add.at(diag,p[kn],1.0); np.add.at(rhs,p[kn],a[y2[kn],x2[kn]])
    A=csr_matrix((vals+list(diag),(rows+list(range(N)),cols+list(range(N)))),shape=(N,N)); sol=np.stack([spsolve(A,rhs[:,c]) for c in range(3)],-1)
    out=a.copy(); fill=a.copy(); fill[ys,xs]=sol; m=np.maximum(hole.astype(float),ndimage.gaussian_filter(ndimage.binary_dilation(hole,iterations=3).astype(float),2.5)*(mn>215))[...,None]; out=out*(1-m)+fill*m
    # the floor around the cut is featureless: smooth it there, so no edge of the filled patch or stray speck shows
    mx2=out.max(-1); mn2=out.min(-1); F=(mn2>215)&((mx2-mn2)/np.maximum(mx2,1)<0.10); zone=ndimage.binary_dilation(hole,iterations=16); zone[:,xc:]=False
    lab,k=ndimage.label(zone&~F&(yy>midline[None,:])); sz=ndimage.sum(np.ones_like(lab),lab,range(1,k+1)); speck=np.isin(lab,1+np.nonzero(sz<200)[0])
    Ff=F.astype(float); wsum=np.maximum(ndimage.gaussian_filter(Ff,7),1e-3); sm=np.stack([ndimage.gaussian_filter(out[...,c]*Ff,7) for c in range(3)],-1)/wsum[...,None]
    w=(ndimage.gaussian_filter(zone.astype(float),4)*(F|ndimage.binary_dilation(speck,iterations=2)))[...,None]; out=out*(1-w)+sm*w
    # the new rounded tip, from the wire's own cross-section just right of the cut
    t,b=T[xc]; prof=a[:,xc-1:xc+2].mean(1); c0=(t+b)/2.0; hh0=(b-t)/2.0+1; yv=np.arange(H); sl=((T[xc+12][0]+T[xc+12][1])-(t+b))/24.0
    for x in range(x_end,xc+1):
        c=c0+sl*(x-xc); f=1-((xc-x)/(half+1.0))**2; hh=hh0*np.sqrt(max(0.0,f))
        if hh<1: continue
        v=(yv-c)/hh; srcy=np.clip(c0+v*hh0,0,H-1); lo=np.floor(srcy).astype(int); hi=np.minimum(lo+1,H-1); fr=(srcy-lo)[:,None]
        col=(prof[lo]*(1-fr)+prof[hi]*fr)*(0.88+0.12*(hh/hh0)); al=(np.clip((1.0-np.abs(v))*hh/1.2,0,1)*(np.abs(v)<=1.0))[:,None]
        out[:,x]=out[:,x]*(1-al)+col*al
    return out
info=json.load(open(D+'/stonefree-frames.json'))
for n in range(1,6):
    I=info[str(n)]; im=Image.open(f'{D}/in/nostone-{n}.png').convert('RGB')
    if list(im.size)!=I['size']: im=im.resize(tuple(I['size']),Image.LANCZOS)
    side=900/I['scale']; canvas=Image.new('RGB',(int(round(side)),int(round(side))),(255,255,255)); canvas.paste(im,(-int(round(I['L'])),-int(round(I['T']))))
    a=np.asarray(canvas.resize((900,900),Image.LANCZOS)).astype(float); H,W=a.shape[:2]; yy,xx=np.mgrid[:H,:W]
    e=np.clip(np.minimum(np.minimum(xx,W-1-xx),np.minimum(yy,H-1-yy))/40.0,0,1); gold=a[...,0]-a[...,2]>35; a=255-(255-a)*np.where(gold,1.0,e)[...,None]
    a=shorten(a,n)
    for metal in ('gold','rose','silver'):
        os.makedirs(f'{OUT}/{metal}',exist_ok=True); px=a if metal=='gold' else tint(a,metal,None)
        Image.fromarray(np.clip(px,0,255).astype(np.uint8)).save(f'{OUT}/{metal}/ring-{n}.jpg',quality=93)
    std=np.asarray(Image.open(f'{D}/../bg/std-{n}.png').convert('RGB')).astype(float); d=np.abs(std-a).max(-1); g=(std[...,0]-std[...,2]>60)
    print(n,'aligned with the supplied photo: median diff on gold',np.median(d[g]))
