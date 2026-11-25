"""Lift one setting. A coil is the same shape at every turn, so what a setting hides is taken from the same spot one
turn up or down (the neighbouring band at the same x), matched to the pixels around the setting. Only the setting's own
footprint is replaced. Where no turn matches (top stones, two-wrap), the row-by-row bridge of lift3 is used."""
import json, os, sys, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import ndimage
from lift3 import bridge
def footprint(shape,cx,cy,r,padL=1.45,padR=1.45,body=1.15,prong=(0.88,0.80,0.78),strip=0.85,extra=(),**_):
    H,W=shape; yy,xx=np.mgrid[:H,:W]; dx=(xx-cx)/r; dy=(yy-cy)/r
    f=dx**2+dy**2<body**2
    for sx in (-1,1):
        for sy in (-1,1): f|=(dx-sx*prong[0])**2+(dy-sy*prong[1])**2<prong[2]**2
    f|=(dx>-padL)&(dx<padR)&(np.abs(dy)<strip)
    for ex,ey,er in extra: f|=(dx-ex)**2+(dy-ey)**2<er**2
    return f
def donor(std,foot,bad,lim=0.25):
    H,W=foot.shape; ys,xs=np.nonzero(foot); ring=ndimage.binary_dilation(foot,iterations=16)&~ndimage.binary_dilation(foot,iterations=3)&~bad
    ry,rx=np.nonzero(ring); tgt=std[ry,rx]; best=None
    for sgn in (-1,1):
        for dy in range(55,135):
            for dx in range(-18,19,2):
                sy=ry+sgn*dy; sx=rx+dx; fy=ys+sgn*dy; fx=xs+dx
                if sy.min()<0 or sy.max()>=H or sx.min()<0 or sx.max()>=W or fy.min()<0 or fy.max()>=H: continue
                if bad[fy,fx].mean()>lim: continue                       # the donor spot must be free of settings itself
                e=np.abs(std[sy,sx]-tgt).mean()
                if best is None or e<best[0]: best=(e,sgn*dy,dx)
    return best
def lift(std,s,kw,bad,top=False,thr=17.0):
    foot=footprint(std.shape[:2],s['x'],s['y'],s['r'],**kw)
    b=None
    if False:
        b=donor(std,foot,bad&~foot)
        if b is not None and b[0]<thr:
            e,dy,dx=b; H,W=foot.shape; src=np.roll(np.roll(std,-dy,0),-dx,1); sbad=np.roll(np.roll(bad&~foot,-dy,0),-dx,1)|np.roll(np.roll(foot,-dy,0),-dx,1)
            ring=ndimage.binary_dilation(foot,iterations=16)&~ndimage.binary_dilation(foot,iterations=3)&~bad
            gain=np.clip(std[ring].mean(0)/np.maximum(src[ring].mean(0),1),0.9,1.12)
            one,m,_=bridge(std,s['x'],s['y'],s['r'],**kw)                 # where the donor spot has a setting of its own, the bridge
            w=ndimage.gaussian_filter((~ndimage.binary_dilation(sbad,iterations=3)).astype(float),2.0)[...,None]
            return one*(1-w)+np.clip(src*gain,0,255)*w,np.maximum(m,ndimage.gaussian_filter(foot.astype(float),1.2)),f'turn dy={dy} dx={dx} err={e:.1f} donor free {1-sbad[foot].mean():.2f}'
    one,m,_=bridge(std,s['x'],s['y'],s['r'],**kw); return one,m,'bridge'
def lift_all(n,std):
    from marks import BEADS
    P=json.load(open('params.json')); R=json.load(open('rounds.json'))[str(n)]; pears=json.load(open('pears.json'))[str(n)]
    from PIL import Image, ImageDraw
    H,W=std.shape[:2]; pm=Image.new('L',(W,H),0); ImageDraw.Draw(pm).polygon([tuple(p) for p in pears['hull']],fill=255); bad=np.asarray(pm)>0
    yy,xx=np.mgrid[:H,:W]
    for x,y,r in BEADS[n]: bad|=(xx-x)**2+(yy-y)**2<(r+4)**2
    kws=[]
    for k,s in enumerate(R):
        kw=dict(P.get(f'{n}-{k}',{})); [kw.pop(q,None) for q in ('x','y','r')]; kws.append(kw); bad|=footprint((H,W),s['x'],s['y'],s['r'],**kw)
    bare=std.copy(); gone=np.zeros((H,W))
    for k,s in enumerate(R):
        one,m,how=lift(std,s,kws[k],bad,top=(k==0)); print('  ',n,k,s['kind'],how)
        bare=bare*(1-m[...,None])+one*m[...,None]; gone=np.maximum(gone,m)
    return bare,gone
if __name__=='__main__':
    from PIL import Image
    for n in range(1,6):
        std=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float); bare,_=lift_all(n,std)
        Image.fromarray(np.clip(bare,0,255).astype(np.uint8)).save(f'nostone-{n}.png')
