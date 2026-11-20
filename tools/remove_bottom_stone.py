"""Take the pear off each ring photo. Behind it: the band is filled from the gold around the gap,
the backdrop from the white around it, and the bottom wire ends in a rounded tip."""
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
pears=json.load(open('pears.json'))
# prong beads, marked by eye on gridded close-ups: tip, top right, bottom right (x, y, r)
BEADS={1:[(328,534,21),(559,455,20),(561,605,21)], 2:[(325,589,21),(553,510,19),(556,650,20)], 3:[(316,604,19),(536,545,20),(535,680,20)],
       4:[(310,668,19),(538,599,20),(540,738,21)], 5:[(321,675,20),(531,608,21),(532,741,21)]}
for n,b in BEADS.items(): pears[str(n)]['beads']=[list(map(float,x)) for x in b]
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
    ring=ndimage.binary_dilation(hole,structure=disk(4))&~hole&g
    ly,lx=np.nonzero(ring&(XX<cxh))
    if len(ly): k=int(np.argmax(ly)); PL=(float(lx[k]),float(ly[k]))
    else: PL=(float(xs.min()),float(y0))
    PR=(float(xR),float(bandR))
    edge=PL[1]+(PR[1]-PL[1])*(np.arange(W)-PL[0])/max(1.0,PR[0]-PL[0])
    bx0,bx1,by0,by1=xs.min()-10,min(W,max(xs.max(),xR)+10),max(0,y0-10),min(H,y1+12)
    sl=(slice(by0,by1),slice(bx0,bx1))
    hs=hole[sl]; above=(YY<=edge[None,:])[sl]; gs=g[sl]; sub=a[sl]
    gd=ndimage.binary_dilation(gs,structure=disk(2))
    # the band behind the stone: carry each row across the gap, following the band's tilt, so its highlights run through
    gold_fill=sub.copy(); hb=hs&above; hh_,ww_=hb.shape; eg=edge[bx0:bx1]
    for yy_ in range(hh_):
        row=hb[yy_]
        if not row.any(): continue
        xs_=np.nonzero(hs[yy_])[0]; segs=np.split(xs_,np.nonzero(np.diff(xs_)>1)[0]+1)
        for sg in segs:
            xa=max(0,sg[0]-3); xb=min(ww_-1,sg[-1]+3); span=max(1,xb-xa)
            for x in sg:
                if not row[x]: continue
                t=(x-xa)/span; ts=t*t*(3-2*t)
                yl=int(round(min(max(yy_+(eg[xa]-eg[x]),0),hh_-1))); yr=int(round(min(max(yy_+(eg[xb]-eg[x]),0),hh_-1)))
                gold_fill[yy_,x]=sub[yl,xa]*(1-ts)+sub[yr,xb]*ts
    gold_fill=np.where(hb[:,:,None],ndimage.gaussian_filter(gold_fill,(1.2,0.6,0)),gold_fill)
    back_fill=harmonic(sub,hs&~above,~gd&~hs,np.array([252.0,251.0,249.0]))
    w=np.clip((edge[None,:]+1.0-YY)/2.0,0,1)[sl][:,:,None]
    fill=a.copy(); fill[sl]=np.where(hs[:,:,None],back_fill*(1-w)+gold_fill*w,sub)
    # the bottom wire carries on a little under where the stone sat and ends in a rounded tip
    colR=a[:,xR:xR+3].mean(1); A=a[:,xR-1:xR+2].mean(1); B=a[:,xR+27:xR+30].mean(1); best=None
    for dd in range(-12,13):
        e=np.abs(A[wt-6:wb+7]-B[wt-6+dd:wb+7+dd]).mean()
        if best is None or e<best[0]: best=(e,dd)
    sw=best[1]/28.0; half=(wb-wt)/2.0+1; cR=(wt+wb)/2.0; yy=np.arange(H)
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
    mask=hole|(wire&(XX<=xR))
    soft=ndimage.gaussian_filter(mask.astype(float),1.2)[:,:,None]
    out=a*(1-soft)+fill*soft
    hx=float(hull[:,0].max()); return out,dict(T=int(T),right=int(hx),wire_at_right=round(float(cR+sw*(hx-xR)),1),tip=[int(x_end),int(round(cR+sw*(x_end-xR)))],wire_half=round(half,1),slope=round(sw,4),body=pg['body'],edge=[PL,PR])
if __name__=='__main__':
    info={}; sheet=Image.new('RGB',(5*440,360),(255,255,255))
    for n in range(1,6):
        a=np.asarray(Image.open(f'nostone-{n}.png').convert('RGB')).astype(float)
        g=ndimage.binary_closing(metal_of(a),structure=np.ones((13,1),bool)); rounds=json.load(open('rounds.json'))[str(n)]; top=rounds[0]
        col=g[:,int(top['x']+3*top['r'])]; T=[r for r in runs(col,minlen=8)][0]; T=int((T[1]-T[0]+1)*1.25)
        T={1:60}.get(n,T); out,inf=erase(a,pears[str(n)],T); info[n]=inf   # only on the 3, 4 and 5 wrap rings does the wire run into the band above
        im=Image.fromarray(np.clip(out,0,255).astype(np.uint8)); im.save(f'bare-{n}.png')
        x0,y0,x1,y1=inf['body']; cx=(x0+x1)//2; cy=(y0+y1)//2
        sheet.paste(im.crop((cx-220,cy-180,cx+220,cy+180)),((n-1)*440,0)); print(n,inf)
    json.dump(info,open('bare.json','w')); sheet.save('bare-sheet.jpg',quality=92)
