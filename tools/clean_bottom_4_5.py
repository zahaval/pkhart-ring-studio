"""Clean the bottom of the Saturday 4- and 5-wrap photos: the grey streaks left in the band where the original pear
was lifted are filled with that band's own gold (row by row, from the gold either side), and the leftover shadow specks
on the floor under it go. Nothing else in the photos is touched."""
import json, os, sys, subprocess, io, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
R='/Users/zahavalowy/Projects/pkhart-ring-studio/'; S=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src=open(R+'tools/make_ring_photos.py').read(); ns={'__file__':R+'tools/make_ring_photos.py'}; exec(src[:src.index('def erase_curved')],ns); tint=ns['tint']; clean=ns['clean_backdrop']
OV={4:(300,596,560,591)}      # the band behind the pear, picked by eye where the automatic pick took the wrong band
PEAR={4:[334,554,232,173],5:[333,597,227,183]}      # where the photo's own pear sat: x, y, w, h
def orig(metal,n): return np.asarray(Image.open(io.BytesIO(subprocess.run(['git','-C',R,'show',f'5ae8774:img/rings/{metal}/ring-{n}.jpg'],capture_output=True).stdout)).convert('RGB')).astype(float)
for n in (4,5):
    a=orig('gold',n); H,W=a.shape[:2]; yy,xx=np.mgrid[:H,:W]; px,py,pw,ph=PEAR[n]
    zone=(xx>px-40)&(xx<px+pw+34)&(yy>py-30)&(yy<py+ph+26)
    mx=a.max(-1); mn=a.min(-1); sat=(mx-mn)/np.maximum(mx,1); lum=a.mean(-1); gold=(sat>0.30)&(a[...,0]-a[...,2]>55)
    up=ndimage.maximum_filter1d(gold.astype(np.uint8),69,axis=0,origin=34)>0; dn=ndimage.maximum_filter1d(gold.astype(np.uint8),69,axis=0,origin=-34)>0
    smear=zone&(sat<0.2)&(lum<238)&up&dn; sy,sx=np.nonzero(smear); ysm=float(np.median(sy)); xsm=float(np.median(sx))
    def runs(x):
        ys=np.nonzero(gold[:,x])[0]; out=[]; s0=p=ys[0]
        for y in ys[1:]:
            if y>p+3: out.append((s0,p)); s0=y
            p=y
        out.append((s0,p)); return [q for q in out if q[1]-q[0]>=8]
    def band(x,yc): return min(runs(x),key=lambda q:abs((q[0]+q[1])/2.0-yc))
    xl=int(sx.min())-10; xr=int(sx.max())+10; yl=yr=ysm
    if n in OV: xl,yl,xr,yr=OV[n]
    bl=band(xl,yl); br=band(xr,yr); bl2=band(xl-20,(bl[0]+bl[1])/2.0); br2=band(xr+20,(br[0]+br[1])/2.0)
    cl=(bl[0]+bl[1])/2.0; cr=(br[0]+br[1])/2.0; sl=(cl-(bl2[0]+bl2[1])/2.0)/20.0; sr=((br2[0]+br2[1])/2.0-cr)/20.0; Wd=float(xr-xl); D=cr-cl
    if n in OV: sl=sr=D/Wd                                  # a straight run between the two sides
    half=max(bl[1]-bl[0],br[1]-br[0])/2.0+5; print(n,'smear at',round(xsm),round(ysm),'band',xl,bl,'->',xr,br,'slopes',round(sl,3),round(sr,3))
    colL=a[:,xl-2:xl+1].mean(1); colR=a[:,xr:xr+3].mean(1); ys_=np.arange(H); f=a.copy(); grey=np.zeros((H,W),bool)
    def samp(col,d):
        q=np.clip(ys_-d,0,H-1); lo=np.floor(q).astype(int); hi=np.minimum(lo+1,H-1); t=(q-lo)[:,None]; return col[lo]*(1-t)+col[hi]*t
    for x in range(xl,xr+1):
        t=(x-xl)/Wd; off=(t**3-2*t*t+t)*sl*Wd+(-2*t**3+3*t*t)*D+(t**3-t*t)*sr*Wd; u=min(1.0,max(0.0,(t-0.5)/0.6+0.5)); ts=u*u*(3-2*u)
        new=samp(colL,off)*(1-ts)+samp(colR,off-D)*ts; rows=np.abs(ys_-(cl+off))<=half; f[rows,x]=new[rows]; grey[rows,x]=True
    mg=ndimage.gaussian_filter(grey.astype(float),1.0)
    left=zone&~grey&(sat<0.2)&(lum<246)&up&dn; left=ndimage.binary_dilation(left,iterations=5)&~ndimage.binary_dilation(gold,iterations=1)&~grey      # streaks left in the gaps go white
    ml=ndimage.gaussian_filter(left.astype(float),1.0)
    # floor specks under the ring, around where the pear was
    mx2=f.max(-1); mn2=f.min(-1); gold2=(((mx2-mn2)/np.maximum(mx2,1))>0.2)&(f[...,0]-f[...,2]>35)|grey      # the ring as it is after the band is rebuilt
    x0,y0,x1,y1=px,py,px+pw,py+ph; low=np.where(gold2,yy,-1).max(0); low=ndimage.maximum_filter1d(low,9)
    lumf=f.mean(-1); speck=(yy>low[None,:]+2)&(xx>x0-90)&(xx<x1+90)&(yy<y1+70)&(lum<240)&(sat<0.2)
    ms=ndimage.gaussian_filter(ndimage.binary_dilation(speck,iterations=6).astype(float),2.0)
    print(n,'band px',int(grey.sum()),'floor px',int(speck.sum()))
    for metal in ('gold','rose','silver'):
        o=orig(metal,n); patch=f if metal=='gold' else clean(tint(f,metal,None))
        out=o*(1-mg[...,None])+patch*mg[...,None]; out=out+(255-out)*np.maximum(ms,ml)[...,None]
        Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save(R+f'img/rings/{metal}/ring-{n}.jpg',quality=93)
sh=Image.new('RGB',(4*640,363),'white')
for i,n in enumerate((4,5)):
    sh.paste(Image.fromarray(orig('gold',n).astype(np.uint8)).crop((80,480,820,900)).resize((640,363)),(i*1280,0)); sh.paste(Image.open(R+f'img/rings/gold/ring-{n}.jpg').crop((80,480,820,900)).resize((640,363)),(i*1280+640,0))
sh.save('v.jpg',quality=93)
