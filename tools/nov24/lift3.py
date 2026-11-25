"""Lift one setting off a ring photo. Only the setting's own footprint (stone, four prongs, the bars along the band) is
replaced, and each row of it is filled from the pixels immediately either side of it, following the band's own bend.
Nothing further away is touched or copied from."""
import json, os, sys, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from scipy import ndimage
def bridge(a,cx,cy,r,padL=1.45,padR=1.45,body=1.15,prong=(0.88,0.80,0.78),strip=0.85,mix=0.35,extra=(),**_):
    H,W,_c=a.shape; xL=int(round(cx-padL*r)); xR=int(round(cx+padR*r)); Wd=float(xR-xL)
    def shift(A,B,lim):
        u0=int(cy-1.0*r); u1=int(cy+1.0*r); best=None
        for dd in range(-lim,lim+1):
            e=np.abs(A[u0:u1]-B[u0+dd:u1+dd]).mean()+0.15*abs(dd)
            if best is None or e<best[0]: best=(e,dd)
        return best[1]
    sl=shift(a[:,xL-25:xL-22].mean(1),a[:,xL-1:xL+2].mean(1),10)/24.0; sr=shift(a[:,xR-1:xR+2].mean(1),a[:,xR+23:xR+26].mean(1),10)/24.0
    D=shift(a[:,xL-2:xL+1].mean(1),a[:,xR:xR+3].mean(1),40); est=(sl+sr)/2*Wd
    if abs(D-est)>10: D=int(round(est))
    def off(x):
        t=(x-xL)/Wd
        if t<=0: return sl*(x-xL)
        if t>=1: return D+sr*(x-xR)
        return (t**3-2*t*t+t)*sl*Wd+(-2*t**3+3*t*t)*D+(t**3-t*t)*sr*Wd
    M=14; x0=xL-M; x1=xR+M; xs=np.arange(x0,x1+1); rel=np.array([off(x) for x in xs]); rel-=off(cx)
    hh=int(1.75*r); ys=np.arange(int(cy)-hh,int(cy)+hh+1)
    # straighten: S[row, col] = the photo at (ys[row]+rel[col], x)
    S=np.zeros((len(ys),len(xs),3))
    for j,x in enumerate(xs):
        yy=np.clip(ys+rel[j],0,H-1); lo=np.floor(yy).astype(int); hi=np.minimum(lo+1,H-1); f=(yy-lo)[:,None]
        S[:,j]=a[lo,x]*(1-f)+a[hi,x]*f
    dx=(xs[None,:]-cx)/r; dy=(ys[:,None]-cy)/r
    foot=dx**2+dy**2<body**2
    for sx in (-1,1):
        for sy in (-1,1): foot|=(dx-sx*prong[0])**2+(dy-sy*prong[1])**2<prong[2]**2
    foot|=(dx>-padL)&(dx<padR)&(np.abs(dy)<strip)
    for ex,ey,er in extra: foot|=(dx-ex)**2+(dy-ey)**2<er**2
    foot[:, :M]=False; foot[:, -M:]=False
    F=S.copy()
    for i in range(len(ys)):
        row=np.nonzero(foot[i])[0]
        if len(row)==0: continue
        for run in np.split(row,np.nonzero(np.diff(row)>1)[0]+1):
            s,e=run[0],run[-1]; A=S[i,max(0,s-3):s].mean(0); B=S[i,e+1:e+4].mean(0); n=e-s+2.0
            t=(run-s+1)/n; m=mix if (e-s)>1.2*r else 1.0; u=np.clip((t-0.5)/m+0.5,0,1); ts=(u*u*(3-2*u))[:,None]
            F[i,run]=A*(1-ts)+B*ts
    Fs=ndimage.gaussian_filter(F,(1.0,0,0)); F=np.where(foot[...,None],Fs,F)
    soft=ndimage.gaussian_filter(foot.astype(float),1.1)
    # back to the photo's own geometry
    out=a.copy(); mask=np.zeros((H,W))
    for j,x in enumerate(xs):
        yy=np.arange(H)-rel[j]-ys[0]; ok=(yy>=0)&(yy<=len(ys)-1); yc=np.clip(yy,0,len(ys)-1); lo=np.floor(yc).astype(int); hi=np.minimum(lo+1,len(ys)-1); f=(yc-lo)
        mcol=(soft[lo,j]*(1-f)+soft[hi,j]*f)*ok; fcol=F[lo,j]*(1-f[:,None])+F[hi,j]*f[:,None]
        out[:,x]=a[:,x]*(1-mcol[:,None])+fcol*mcol[:,None]; mask[:,x]=mcol
    return out,mask,(sl,sr,D)
if __name__=='__main__':
    from lift_all import ROUNDS
    P=json.load(open('params.json')) if os.path.exists('params.json') else {}; outj={}
    for n,st in ROUNDS.items():
        std=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float); bare=std.copy(); outj[str(n)]=[]
        for k,(kind,x,y,r) in enumerate(st):
            kw=dict(P.get(f'{n}-{k}',{})); x=kw.pop('x',x); y=kw.pop('y',y); r=kw.pop('r',r)
            one,m,info=bridge(std,x,y,r,**kw); print(n,k,kind,[round(float(v),3) for v in info])
            bare=bare*(1-m[...,None])+one*m[...,None]; outj[str(n)].append(dict(kind=kind,x=x,y=y,r=r,wire_y=y))
        Image.fromarray(np.clip(bare,0,255).astype(np.uint8)).save(f'nostone-{n}.png')
    json.dump(outj,open('rounds.json','w'))
