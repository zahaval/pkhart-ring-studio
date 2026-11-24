import json, os, sys, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw
from scipy import ndimage
from lift_all import ROUNDS
def bridge(a,cx,cy,r,padL=1.45,padR=1.45,hy=1.34,feather=4.0,mix=0.35,body=1.1,prong=(0.88,0.78,0.7),cap=None):
    """rebuild what is under one setting from the columns just left and right of it, following the wire's own bend"""
    H,W,_=a.shape; xL=int(round(cx-padL*r)); xR=int(round(cx+padR*r)); half=hy*r; Wd=float(xR-xL)
    colL=a[:,xL-2:xL+1].mean(1); colR=a[:,xR:xR+3].mean(1)
    def shift(A,B,lim):
        u0=int(cy-1.0*r); u1=int(cy+1.0*r); best=None
        for dd in range(-lim,lim+1):
            e=np.abs(A[u0:u1]-B[u0+dd:u1+dd]).mean()+0.15*abs(dd)
            if best is None or e<best[0]: best=(e,dd)
        return best[1]
    sl=shift(a[:,xL-25:xL-22].mean(1),a[:,xL-1:xL+2].mean(1),10)/24.0; sr=shift(a[:,xR-1:xR+2].mean(1),a[:,xR+23:xR+26].mean(1),10)/24.0
    D=shift(colL,colR,40); est=(sl+sr)/2*Wd
    if abs(D-est)>10: D=int(round(est))
    out=a.copy(); ys=np.arange(H)
    for x in range(xL,xR+1):
        t=(x-xL)/Wd; u=min(1.0,max(0.0,(t-0.5)/mix+0.5)); ts=u*u*(3-2*u)      # each side's own band runs in to the middle; they meet under the stone
        off=(t**3-2*t*t+t)*sl*Wd+(-2*t**3+3*t*t)*D+(t**3-t*t)*sr*Wd
        def samp(col,d):
            yy=np.clip(ys-d,0,H-1); lo=np.floor(yy).astype(int); hi=np.minimum(lo+1,H-1); f=(yy-lo)[:,None]
            return col[lo]*(1-f)+col[hi]*f
        A=samp(colL,off); B=samp(colR,off-D); new=A*(1-ts)+B*ts
        c=cy+off-D/2.0
        if cap:                                # above and below the band, copy only from the side away from the ring's edge
            wc=np.clip((np.abs(ys-c)/r-0.75)/0.12,0,1)[:,None]; new=new*(1-wc)+(B if cap=='R' else A)*wc
        w=np.clip((half-np.abs(ys-c))/feather,0,1)[:,None]
        out[:,x]=a[:,x]*(1-w)+new*w
    # only what the setting itself covered is replaced: the stone, its four prongs and the little gallery on the side
    # facing the middle of the ring. The band and gaps around them stay exactly as photographed.
    yy,xx=np.mgrid[:H,:W]; dx=(xx-cx)/r; dy=(yy-cy)/r
    foot=dx**2+dy**2<body**2
    for sx in (-1,1):
        for sy in (-1,1): foot|=(dx-sx*prong[0])**2+(dy-sy*prong[1])**2<prong[2]**2
    foot|=(dx>-padL)&(dx<padR)&(np.abs(dy)<0.85)                 # the bars and gallery either side of the stone, along the band
    m=(np.abs(out-a).max(-1)>0)&foot; m=ndimage.gaussian_filter(m.astype(float),1.0)
    return out,m,(sl,sr,D)
if __name__=='__main__':
    P=json.load(open('params.json')) if os.path.exists('params.json') else {}
    R=json.load(open('../m2/rounds.json')); outj={'1':R['1']}
    for n,st in ROUNDS.items():
        std=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float); bare=std.copy(); outj[str(n)]=[]
        for k,(kind,x,y,r) in enumerate(st):
            kw=P.get(f'{n}-{k}',{}); x=kw.pop('x',x); y=kw.pop('y',y); r=kw.pop('r',r)
            one,m,info=bridge(std,x,y,r,**kw); print(n,k,kind,[round(float(v),3) for v in info])
            bare=bare*(1-m[...,None])+one*m[...,None]; outj[str(n)].append(dict(kind=kind,x=x,y=y,r=r,wire_y=y))
        Image.fromarray(np.clip(bare,0,255).astype(np.uint8)).save(f'nostone-{n}.png')
    json.dump(outj,open('rounds.json','w'))
