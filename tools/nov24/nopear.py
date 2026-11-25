"""Take the pear off each ring photo. Behind it: the band is filled from the gold around the gap,
the backdrop from the white around it, and the bottom wire ends in a rounded tip."""
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import sys
pears=json.load(open('pears.json')); TT=json.load(open('T.json')) if __import__('os').path.exists('T.json') else {}; TT={int(k):v for k,v in TT.items()}
# prong beads, marked by eye on gridded close-ups: tip, top right, bottom right (x, y, r)
import os,sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marks import BEADS
for n,b in BEADS.items(): pears[str(n)]['beads']=[list(map(float,x)) for x in b]
OV=json.load(open('wire.json')) if os.path.exists('wire.json') else {}
for n,o in OV.items(): pears[n].update(o)
disk=lambda q:(np.hypot(*np.mgrid[-q:q+1,-q:q+1])<=q)
def gold_of(a):
    r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]; mx=a.max(2); mn=a.min(2); sat=(mx-mn)/np.maximum(mx,1)
    return (r-b>45)&(sat>0.28)&(r>110)
def metal_of(a):
    """anything that is wire: gold, its dark reflections and its pale highlights (the backdrop and its shadows are grey-white)"""
    mx=a.max(2); mn=a.min(2); sat=(mx-mn)/np.maximum(mx,1); lum=0.3*a[:,:,0]+0.59*a[:,:,1]+0.11*a[:,:,2]
    return (sat>0.17)|(lum<150)
def runs(col,minlen=14,gap=3):
    ys=np.nonzero(col)[0]
    if len(ys)==0: return []
    out=[]; s=ys[0]; p=ys[0]
    for y in ys[1:]:
        if y>p+gap: out.append((s,p)); s=y
        p=y
    out.append((s,p)); return [r for r in out if r[1]-r[0]>=minlen]
def harmonic(img,fillm,usable,init,iters=1500):
    """fill `fillm` by spreading in the values of the `usable` pixels around it"""
    sub=img.copy(); sub[fillm]=init; ok=(fillm|usable).astype(float); kern=np.array([[0,1,0],[1,0,1],[0,1,0]],float)
    for it in range(iters):
        num=np.stack([ndimage.convolve(sub[:,:,c]*ok,kern,mode='nearest') for c in range(3)],2)
        den=ndimage.convolve(ok,kern,mode='nearest')[:,:,None]
        sub[fillm]=(num/np.maximum(den,1e-6))[fillm]
    return sub
def erase(a,pg,T):
    H,W,_=a.shape; hull=np.array(pg['hull']); beads=pg['beads']
    m=Image.new('L',(W,H),0); ImageDraw.Draw(m).polygon([tuple(p) for p in hull.tolist()],fill=255); hole=np.asarray(m)>0
    YY,XX=np.mgrid[0:H,0:W]
    for x,y,r in beads: hole|=(XX-x)**2+(YY-y)**2<=(r+5)**2
    hole=ndimage.binary_dilation(hole,structure=disk(5))
    g=gold_of(a); gc=ndimage.binary_closing(metal_of(a),structure=np.ones((13,1),bool))   # gc: highlights along a wire bridged, for measuring runs
    gl,gk=ndimage.label(g&~hole); gsz=ndimage.sum(g&~hole,gl,range(1,gk+1))
    scraps=np.isin(gl,1+np.nonzero(gsz<1500)[0])&ndimage.binary_dilation(hole,structure=disk(14))   # bits of prong left beside it
    hole|=ndimage.binary_dilation(scraps,structure=disk(3))
    g2=g&~hole
    ys,xs=np.nonzero(hole); y0=ys.min(); y1=ys.max(); cxh=(hull[:,0].min()+hull[:,0].max())/2.0
    xR=int(max(hull[:,0].max(),max(b[0]+b[2] for b in beads)))+12
    rr=runs((gc&~hole)[:,xR]); wt,wb=rr[-1]; print('   runs at',xR,rr,'T',T)
    if T and wb-wt>T+10: bandR=wb-T; wt=wb-T+1                          # the wire touches the band above here
    else: bandR=rr[-2][1] if len(rr)>1 else wt-1
    if pg.get('wire'): wt,wb,bandR=pg['wire']                          # measured by eye where a highlight splits the wire
    # the band behind the stone: its profile just left of the stone and just right of it, carried straight across with
    # the band's own tilt, so its edges and highlights run through as clean lines; under its lower edge, plain floor
    XL=int(xs.min())-5; yb=min(b[1] for b in beads[:1])
    rl=[r for r in runs((gc&~hole)[:,XL],minlen=8) if r[1]<yb+6]; edgeL=pg.get('edgeL',rl[-1][1] if rl else y0); edgeR=bandR
    print('   left runs at',XL,rl,'edgeL',edgeL,'edgeR',edgeR)
    colL=a[:,XL-2:XL+1].mean(1); colRR=a[:,xR:xR+3].mean(1); D=float(edgeR-edgeL); Wd=float(xR-XL); yr=np.arange(H); fill=a.copy()
    WHITE=np.array([254.0,254.0,254.0])
    def samp(col,d):
        q=np.clip(yr-d,0,H-1); lo=np.floor(q).astype(int); hi=np.minimum(lo+1,H-1); f=(q-lo)[:,None]
        return col[lo]*(1-f)+col[hi]*f
    def tilt(x0,x1,ye):                                      # the band's own slope at one side, from its profile 24px apart
        A=a[:,x0-1:x0+2].mean(1); B=a[:,x1-1:x1+2].mean(1); u0=int(ye-70); u1=int(ye+12); best=None
        for dd in range(-10,11):
            e=np.abs(A[u0:u1]-B[u0+dd:u1+dd]).mean()+0.1*abs(dd)
            if best is None or e<best[0]: best=(e,dd)
        return best[1]/float(x1-x0)
    sL=pg.get('sL',tilt(XL-24,XL,edgeL)); sR=pg.get('sR',tilt(xR,xR+24,edgeR)); print('   tilts',round(sL,3),round(sR,3),'rise',D)
    def offs(x):                                             # the band follows a smooth arc between the two sides
        t=(x-XL)/Wd; return (t**3-2*t*t+t)*sL*Wd+(-2*t**3+3*t*t)*D+(t**3-t*t)*sR*Wd
    OFF=np.array([offs(x) for x in range(W)])
    for x in range(XL,xR+1):
        t=(x-XL)/Wd; u=min(1.0,max(0.0,(t-0.5)/0.6+0.5)); ts=u*u*(3-2*u); off=OFF[x]
        new=samp(colL,off)*(1-ts)+samp(colRR,off-D)*ts
        ww=np.clip(yr-(edgeL+off)+0.5,0,1)[:,None]
        fill[:,x]=new*(1-ww)+WHITE*ww
    PL=(float(XL),float(edgeL)); PR=(float(xR),float(edgeR))
    # the stone's point hangs below the band, so above the point the band is only hidden from where the stone's body
    # starts (XL2): left of that the band stays exactly as photographed, and the rebuilt part is as short as it can be
    tipx,tipy,tipr=beads[0]; ysplit=tipy-tipr-6; rowsS=YY-OFF[None,:]
    hx=np.nonzero((hole&(rowsS<ysplit)&(XX>tipx)).any(0))[0]; XL2=int(pg.get('XL2',(hx.min()-8) if len(hx) else XL)); XL2=max(XL2,XL)
    print('   XL2',XL2,'ysplit',round(ysplit))
    if XL2>XL+10:
        colL2=a[:,XL2-2:XL2+1].mean(1); W2=float(xR-XL2); fill2=fill.copy()
        for x in range(XL2,xR+1):
            t=(x-XL2)/W2; u=min(1.0,max(0.0,(t-0.5)/0.6+0.5)); ts=u*u*(3-2*u)
            new=samp(colL2,OFF[x]-OFF[XL2])*(1-ts)+samp(colRR,OFF[x]-D)*ts
            ww=np.clip(yr-(edgeL+OFF[x])+0.5,0,1)[:,None]; fill2[:,x]=new*(1-ww)+WHITE*ww
        up=np.clip((ysplit-rowsS)/6.0,0,1)[...,None]                        # 1 above the point, 0 at and below it
        fill2=np.where((XX<XL2)[...,None],a,fill2); fill=fill*(1-up)+fill2*up
    else: XL2=XL
    topL=pg.get('topL',rl[-1][0] if rl else y0)                      # that band and the floor under it are rebuilt whole, so no scrap of a prong is left on them
    rect=(XX>=XL)&(XX<=xR)&(YY>=topL+OFF[None,:]-3)&(YY<=y1+12)
    rect&=(XX>=XL2-2)                                  # left of the stone's body only the stone's own point is replaced
    # the bottom wire carries on a little under where the stone sat and ends in a rounded tip
    colR=a[:,xR:xR+3].mean(1); A=a[:,xR-1:xR+2].mean(1); B=a[:,xR+27:xR+30].mean(1); best=None
    for dd in range(-12,13):
        e=np.abs(A[wt-6:wb+7]-B[wt-6+dd:wb+7+dd]).mean()
        if best is None or e<best[0]: best=(e,dd)
    sw=pg.get('slope',best[1]/28.0); half=(wb-wt)/2.0+1; cR=(wt+wb)/2.0; yy=np.arange(H)
    x_end=int(hull[:,0].max())-int(round(0.19*(hull[:,0].max()-hull[:,0].min()))); wire=np.zeros((H,W),bool)
    for x in range(x_end,xR+1):
        c=cR+sw*(x-xR)
        hh=half if x>=x_end+half else half*np.sqrt(max(0.0,1-((x_end+half-x)/half)**2))
        if hh<1: continue
        v=(yy-c)/hh; srcy=np.clip(cR+v*half,0,H-1); lo=np.floor(srcy).astype(int); hi=np.minimum(lo+1,H-1); f=(srcy-lo)[:,None]
        col=colR[lo]*(1-f)+colR[hi]*f
        if x<x_end+half: col=col*(0.86+0.14*(hh/half))
        al=(np.clip((1.0-np.abs(v))*hh/1.2,0,1)*(np.abs(v)<=1.0))[:,None]
        fill[:,x]=fill[:,x]*(1-al)+col*al; wire[:,x]=np.abs(v)<=1.0
    mask=hole|rect|(wire&(XX<=xR))
    soft=ndimage.gaussian_filter(mask.astype(float),1.2)[:,:,None]
    out=a*(1-soft)+fill*soft
    hx=float(hull[:,0].max()); return out,dict(T=int(T),right=int(hx),wire_at_right=round(float(cR+sw*(hx-xR)),1),tip=[int(x_end),int(round(cR+sw*(x_end-xR)))],wire_half=round(half,1),slope=round(sw,4),body=pg['body'],edge=[PL,PR])
if __name__=='__main__':
    info={}; sheet=Image.new('RGB',(5*600,500),(255,255,255))
    for n in range(1,6):
        a=np.asarray(Image.open(f'nostone-{n}.png').convert('RGB')).astype(float)
        g=ndimage.binary_closing(metal_of(a),structure=np.ones((13,1),bool)); rounds=json.load(open('rounds.json'))[str(n)]; top=rounds[0]
        col=g[:,int(top['x']+3*top['r'])]; T=[r for r in runs(col,minlen=8)][0]; T=int((T[1]-T[0]+1)*1.25)
        T=TT.get(n,T); out,inf=erase(a,pears[str(n)],T); info[n]=inf   # only on the 3, 4 and 5 wrap rings does the wire run into the band above
        im=Image.fromarray(np.clip(out,0,255).astype(np.uint8)); im.save(f'bare-{n}.png')
        x0,y0,x1,y1=inf['body']; cx=(x0+x1)//2; cy=(y0+y1)//2
        sheet.paste(im.crop((cx-300,cy-250,cx+300,cy+250)),((n-1)*600,0)); print(n,inf)
    json.dump(info,open('bare.json','w')); sheet.save('bare-sheet.jpg',quality=92)
