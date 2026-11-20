"""Lift the round stones off each ring: rebuild the wire under a stone from the wire either side of it."""
import json
import numpy as np
from PIL import Image
from scipy import ndimage
found=json.load(open('found.json'))
def goldmask(a):
    r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]; mx=a.max(2); mn=a.min(2); sat=(mx-mn)/np.maximum(mx,1)
    return (r-b>45)&(sat>0.28)&(r>110)
def run_at(g,x,yc,maxgap=3):
    """the gold run in column x nearest to yc: (top, bottom)"""
    col=g[:,x]; ys=np.nonzero(col)[0]
    if len(ys)==0: return None
    runs=[]; s=ys[0]; p=ys[0]
    for y in ys[1:]:
        if y>p+maxgap: runs.append((s,p)); s=y
        p=y
    runs.append((s,p))
    runs=[r for r in runs if r[1]-r[0]>=12]
    if not runs: return None
    return min(runs,key=lambda r:abs((r[0]+r[1])/2-yc))
def erase_round(a,cx,cy,r):
    H,W,_=a.shape
    xL=int(round(cx-1.8*r)); xR=int(round(cx+1.8*r))
    half=int(1.5*r)
    colL=a[:,xL-2:xL+1].mean(1); colR=a[:,xR:xR+3].mean(1)
    # the wire's slope just left and just right of the setting (same wire, 28px apart)
    def local_shift(x0,x1):
        A=a[:,x0-1:x0+2].mean(1); B=a[:,x1-1:x1+2].mean(1); u0=int(cy-1.0*r); u1=int(cy+1.0*r); best=None
        for dd in range(-12,13):
            e=np.abs(A[u0:u1]-B[u0+dd:u1+dd]).mean()
            if best is None or e<best[0]: best=(e,dd)
        return best[1]/float(x1-x0)
    sl=local_shift(xL-28,xL); sr=local_shift(xR,xR+28)
    d=(sl+sr)/2.0*(xR-xL)
    # refine the rise across the gap by matching the two edge profiles directly, near the slope's guess
    u0=int(cy-1.3*r); u1=int(cy+1.3*r); best=None
    for dd in range(int(round(d))-9,int(round(d))+10):
        e=np.abs(colL[u0:u1]-colR[u0+dd:u1+dd]).mean()+0.15*abs(dd-d)
        if best is None or e<best[0]: best=(e,dd)
    d=float(best[1]); cL=cy-d/2.0; cR=cy+d/2.0
    m0=sl*(xR-xL); m1=sr*(xR-xL)
    out=a.copy(); ys=np.arange(H)
    for x in range(xL,xR+1):
        t=(x-xL)/(xR-xL); ts=t*t*(3-2*t)
        c=cL+(t**3-2*t*t+t)*m0+(-2*t**3+3*t*t)*d+(t**3-t*t)*m1     # the wire's path: leaves and arrives at the wire's own slope
        def samp(col,c0):
            yy=np.clip(ys-(c-c0),0,H-1); lo=np.floor(yy).astype(int); hi=np.minimum(lo+1,H-1); f=(yy-lo)[:,None]
            return col[lo]*(1-f)+col[hi]*f
        new=samp(colL,cL)*(1-ts)+samp(colR,cR)*ts
        w=(np.clip((half-np.abs(ys-c))/8.0,0,1)*min(1.0,(x-xL+1)/8.0,(xR-x+1)/8.0))[:,None]
        out[:,x]=a[:,x]*(1-w)+new*w
    return out,(xL,xR,cL,cR)
if __name__=='__main__':
    sheet=Image.new('RGB',(5*600,600),(255,255,255)); geo={}
    for n in range(1,6):
        a=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float)
        rounds=[f for f in found[str(n)] if f['w']<130 and f['h']>50 and abs(f['w']-f['h'])<25]
        rounds.sort(key=lambda f:f['y']); geo[n]=[]
        for f in rounds:
            r=(f['w']+f['h'])/4
            a,(xL,xR,cL,cR)=erase_round(a,f['x'],f['y'],r)
            geo[n].append(dict(kind=f['kind'],x=round(f['x']),y=round(f['y']),r=round(r,1),wire_y=round(cL+(cR-cL)*(f['x']-xL)/(xR-xL),1)))
        im=Image.fromarray(np.clip(a,0,255).astype(np.uint8)); im.save(f'nostone-{n}.png')
        sheet.paste(im.resize((600,600),Image.LANCZOS),((n-1)*600,0)); print(n,geo[n])
    json.dump(geo,open('rounds.json','w')); sheet.save('nostone-sheet.jpg',quality=90)
