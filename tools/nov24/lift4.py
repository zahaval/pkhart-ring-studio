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
def true_foot(shape,s,balls,inner):
    """the setting's own outline: the stone, its four prong balls, the bars between them and the little gallery"""
    H,W=shape; yy,xx=np.mgrid[:H,:W]; x,y,r=s['x'],s['y'],s['r']
    f=(xx-x)**2+(yy-y)**2<(1.04*r+3)**2
    for bx,by,br,ok in balls: f|=(xx-bx)**2+(yy-by)**2<(br+6)**2
    for a,b in ((0,1),(2,3),(0,2),(1,3)):                      # bars: a capsule between neighbouring balls
        (ax,ay,ar,_),(bx,by,br,_)=balls[a],balls[b]; t=np.clip(((xx-ax)*(bx-ax)+(yy-ay)*(by-ay))/max(1.0,(bx-ax)**2+(by-ay)**2),0,1)
        f|=(xx-(ax+t*(bx-ax)))**2+(yy-(ay+t*(by-ay)))**2<(0.62*max(ar,br)+2)**2
    gx=(balls[2][0]+balls[3][0])/2 if inner>0 else (balls[0][0]+balls[1][0])/2
    f|=((xx-(gx+inner*0.22*r))/(0.42*r))**2+((yy-y)/(0.58*r))**2<1
    return f
def lift_all(n,std):
    """every setting lifted by the row-by-row bridge, its outline kept inside the box of its own four prong balls"""
    R=json.load(open('rounds.json'))[str(n)]; BL=json.load(open('balls.json'))[str(n)]; P=json.load(open('params.json'))
    H,W=std.shape[:2]; bare=std.copy(); gone=np.zeros((H,W))
    for k,s in enumerate(R):
        kw=dict(P.get(f'{n}-{k}',{})); [kw.pop(q,None) for q in ('x','y','r')]; x,y,r=s['x'],s['y'],s['r']; b=BL[k]; g=7.0
        cl=(x-min(q[0]-q[2] for q in b)+g)/r; cr=(max(q[0]+q[2] for q in b)-x+g)/r; ct=(y-min(q[1]-q[2] for q in b)+g)/r; cb=(max(q[1]+q[2] for q in b)-y+g)/r
        if k>0:                                                      # the gallery sticks out a little on the side facing the middle
            if x<450: cr=max(cr,kw.get('padR',1.45))
            else: cl=max(cl,kw.get('padL',1.45))
        if k==0: cl,cr=max(cl,1.42),max(cr,1.42)
        ct,cb=max(ct,1.32),max(cb,1.32)      # a missed ball must still be inside
        cl,cr,ct,cb=[min(v,1.9) for v in (cl,cr,ct,cb)]
        one,m,_=bridge(std,x,y,r,clip=(cl,cr,ct,cb),**kw); print('  ',n,k,s['kind'],'clip',[round(v,2) for v in (cl,cr,ct,cb)])
        bare=bare*(1-m[...,None])+one*m[...,None]; gone=np.maximum(gone,m)
        if k>0:
            # the prongs sat over the gaps and the edges of the bands above and below: those small caps are filled in 2D
            # from everything around them (so an edge that crosses a cap carries on through it), not along the band
            from skimage.restoration import inpaint_biharmonic
            yy,xx=np.mgrid[:H,:W]; Wd=(kw.get('padL',1.45)+kw.get('padR',1.45))*r; dyb=yy-y-_[2]*(xx-x)/Wd
            cap=ndimage.binary_dilation((m>0.02)&(np.abs(dyb)>0.52*r),iterations=2)
            ys_,xs_=np.nonzero(cap); y0,y1,x0,x1=max(0,ys_.min()-12),min(H,ys_.max()+13),max(0,xs_.min()-12),min(W,xs_.max()+13)
            sub=inpaint_biharmonic(bare[y0:y1,x0:x1]/255.0,cap[y0:y1,x0:x1],channel_axis=-1)*255.0
            w=ndimage.gaussian_filter(cap[y0:y1,x0:x1].astype(float),0.8)[...,None]; bare[y0:y1,x0:x1]=bare[y0:y1,x0:x1]*(1-w)+np.clip(sub,0,255)*w
            gone=np.maximum(gone,cap.astype(float))
    return bare,gone
if __name__=='__main__':
    from PIL import Image
    for n in range(1,6):
        std=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float); bare,_=lift_all(n,std)
        Image.fromarray(np.clip(bare,0,255).astype(np.uint8)).save(f'nostone-{n}.png')
