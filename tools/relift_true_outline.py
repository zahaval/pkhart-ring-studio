"""Lift each of the photo's own settings by its true outline only (stone, four prong balls, the bars between them, the
little gallery), filled row by row from the band right beside it. Everything outside that outline is the photo itself.
The bottom-stone area is taken from the version already live. Background = the photo's own."""
import json, os, sys, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
PAD=float(sys.argv[1]) if len(sys.argv)>1 else 1.7
R='/Users/zahavalowy/Projects/pkhart-ring-studio/'; D=os.path.dirname(os.path.abspath(__file__))
src=open(R+'tools/make_ring_photos.py').read(); ns={'__file__':R+'tools/make_ring_photos.py'}; exec(src[:src.index("rounds=json.load(open('rounds.json'))")],ns); tint=ns['tint']; erase_curved=ns['erase_curved']
esrc=open(D+'/erase_mod.py').read(); ens={}; exec(esrc[:esrc.index("if __name__")].replace("found=json.load(open('found.json'))",''),ens); erase_round=ens['erase_round']
ST={1:[(446,327,54.5)],2:[(436,304,39.8),(278,444,40.0)],3:[(418,226,37.8),(274,368,40.5),(541,454,41.0)],
    4:[(416,219,44.0),(261,334,42.0),(537,415,42.8),(325,521,35.8)],5:[(431,164,45.8),(262,282,42.8),(542,367,45.2),(328,475,38.2),(649,556,34.8)]}
BR={1:(311,463,625,441),2:(304,547,608,559),3:(290,539,603,540)}      # the band behind the pear: x and band centre, left and right of it
PEAR={1:[351,454,238,179],2:[344,548,228,177],3:[330,535,237,177],4:[334,554,232,173],5:[333,597,227,183]}
def balls(gray,x,y,r):
    ox,oy=max(0,int(x-1.9*r)),max(0,int(y-1.9*r)); crop=gray[oy:int(y+1.9*r),ox:int(x+1.9*r)]
    ed=canny(crop,sigma=1.4,low_threshold=8,high_threshold=25); radii=np.arange(max(5,int(0.22*r)),int(0.42*r)+1)
    acc,cxs,cys,rs=hough_circle_peaks(hough_circle(ed,radii),radii,min_xdistance=int(0.3*r),min_ydistance=int(0.3*r),total_num_peaks=40,normalize=True)
    cs=[(cx+ox,cy+oy,rr,ac) for ac,cx,cy,rr in zip(acc,cxs,cys,rs)]; out=[]
    for sx in (-1,1):
        for sy in (-1,1):
            ex,ey=x+sx*0.9*r,y+sy*0.85*r; cand=[c for c in cs if np.hypot(c[0]-ex,c[1]-ey)<0.4*r and (c[0]-x)*sx>0.45*r and (c[1]-y)*sy>0.4*r]
            if cand: c=max(cand,key=lambda c:c[3]-0.004*np.hypot(c[0]-ex,c[1]-ey)); out.append((float(c[0]),float(c[1]),float(c[2])))
            else: out.append((ex,ey,0.3*r))
    return out
def outline(shape,x,y,r,bl,inner):
    H,W=shape; yy,xx=np.mgrid[:H,:W]; f=(xx-x)**2+(yy-y)**2<(1.0*r+3)**2
    for bx,by,br in bl: f|=(xx-bx)**2+(yy-by)**2<(max(br,0.3*r)*1.25+5)**2
    for a,b in ((0,1),(2,3),(0,2),(1,3)):
        (ax,ay,ar),(bx,by,br)=bl[a],bl[b]; t=np.clip(((xx-ax)*(bx-ax)+(yy-ay)*(by-ay))/max(1.0,(bx-ax)**2+(by-ay)**2),0,1)
        f|=(xx-(ax+t*(bx-ax)))**2+(yy-(ay+t*(by-ay)))**2<(0.7*max(ar,br)+2)**2
    gx=max(b[0] for b in bl) if inner>0 else min(b[0] for b in bl)
    f|=((xx-(gx+inner*0.2*r))/(0.5*r))**2+((yy-y)/(0.62*r))**2<1
    return f
def build(n,dbg=None):
    std=np.asarray(Image.open(f'{D}/std-{n}.png').convert('RGB')).astype(float); live=np.asarray(Image.open(f'{D}/live/ring-{n}.jpg').convert('RGB')).astype(float)
    gray=std.mean(-1); out=std.copy(); H,W=gray.shape
    for k,(x,y,r) in enumerate(ST[n]):
        bl=balls(gray,x,y,r); inner=1 if x<450 else -1; F=outline((H,W),x,y,r,bl,inner if k>0 else 1)
        if k==0: F|=outline((H,W),x,y,r,bl,-1)
        res=erase_curved(std.copy(),x,y,r,pad=PAD,halfk=1.7)          # the band carried across from clear of the setting on both sides
        one=res[0] if res is not None else erase_round(std.copy(),x,y,r)[0]
        m=ndimage.gaussian_filter(ndimage.binary_dilation(F,iterations=5).astype(float),2.2)
        out=out*(1-m[...,None])+one*m[...,None]
    px,py,pw,ph=PEAR[n]; yy,xx=np.mgrid[:H,:W]; zone=(xx>px-70)&(xx<px+pw+70)&(yy>py-40)&(yy<py+ph+110)
    pm=ndimage.binary_dilation(ndimage.binary_closing((np.abs(live-std).max(-1)>14)&zone,iterations=3),iterations=3)&zone
    pbox=(xx>px-30)&(xx<px+pw+30)&(yy>py-26)&(yy<py+ph+30)
    for (sx,sy,sr) in ST[n]: pm&=~((np.abs(xx-sx)<1.95*sr)&(np.abs(yy-sy)<1.65*sr)&~pbox)      # not the old wide patch around a stone that sits near the pear
    lab,k2=ndimage.label(pm); sz=ndimage.sum(pm,lab,range(1,k2+1)); pm=np.isin(lab,1+np.nonzero(sz>3000)[0])       # the pear's own area, not specks
    pmf=ndimage.gaussian_filter(pm.astype(float),1.2)[...,None]; out=out*(1-pmf)+live*pmf
    if n in BR:
        # the band the pear sat in front of: rebuilt from that band just left and just right of the pear, so it runs
        # through with clean edges; any torn scraps just under it become floor
        xl,yl,xr,yr=BR[n]; mx2=out.max(-1); mn2=out.min(-1); g2=((mx2-mn2)/np.maximum(mx2,1)>0.28)&(out[...,0]-out[...,2]>50)
        def band(x,yc):
            ys=np.nonzero(g2[:,x])[0]; rr=[]; s0=p=ys[0]
            for y in ys[1:]:
                if y>p+3: rr.append((s0,p)); s0=y
                p=y
            rr.append((s0,p)); return min([q for q in rr if q[1]-q[0]>=6],key=lambda q:abs((q[0]+q[1])/2.0-yc))
        bl,br=band(xl,yl),band(xr,yr); cl=(bl[0]+bl[1])/2.0; cr=(br[0]+br[1])/2.0; Wd=float(xr-xl); Dd=cr-cl; hl=(bl[1]-bl[0])/2.0; hr=(br[1]-br[0])/2.0
        colL=out[:,xl-2:xl+1].mean(1); colR=out[:,xr:xr+3].mean(1); yv=np.arange(H); print('  band',n,bl,'->',br)
        def samp(col,d):
            q=np.clip(yv-d,0,H-1); lo=np.floor(q).astype(int); hi=np.minimum(lo+1,H-1); t=(q-lo)[:,None]; return col[lo]*(1-t)+col[hi]*t
        for x in range(xl,xr+1):
            t=(x-xl)/Wd; off=Dd*t; u=min(1.0,max(0.0,(t-0.5)/0.6+0.5)); ts=u*u*(3-2*u); half=hl*(1-t)+hr*t+4
            new=samp(colL,off)*(1-ts)+samp(colR,off-Dd)*ts; c=cl+off; rows=np.abs(yv-c)<=half; col=out[:,x].copy(); col[rows]=new[rows]
            below=(yv>c+half)&(yv<=c+half+18)&(col[:,0]-col[:,2]>30)&(yv<py+ph); fl=out[min(H-1,int(c+half+24)):min(H,int(c+half+28)),x].mean(0)
            if (fl[0]-fl[2])<25: col[below]=fl
            out[:,x]=col
    e=np.clip(np.minimum(np.minimum(xx,W-1-xx),np.minimum(yy,H-1-yy))/40.0,0,1); gold=(out[...,0]-out[...,2]>35); out=255-(255-out)*np.where(gold,1.0,e)[...,None]
    return std,out
if __name__=='__main__':
    tiles=[]
    for n in range(1,6):
        std,out=build(n)
        for metal in ('gold','rose','silver'):
            px=out if metal=='gold' else tint(out,metal,None); Image.fromarray(np.clip(px,0,255).astype(np.uint8)).save(R+f'img/rings/{metal}/ring-{n}.jpg',quality=93)
        sil=Image.open(R+f'img/rings/silver/ring-{n}.jpg'); o=Image.fromarray(std.astype(np.uint8))
        for (x,y,r) in ST[n]:
            b=(int(max(0,x-150)),int(y-80),int(max(0,x-150))+300,int(y+80)); tiles.append((o.crop(b).resize((450,240),Image.LANCZOS),sil.crop(b).resize((450,240),Image.LANCZOS)))
    sh=Image.new('RGB',(4*910,4*245),'white')
    for i,(a,b) in enumerate(tiles): sh.paste(a,((i%4)*910,(i//4)*245)); sh.paste(b,((i%4)*910+455,(i//4)*245))
    sh.save(D+'/silver-tiles.jpg',quality=90)
