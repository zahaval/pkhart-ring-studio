import json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull
F=json.load(open('found.json'))
def pear_geom(n):
    a=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(int)
    r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]; mx=a.max(2); mn=a.min(2); sat=(mx-mn)/np.maximum(mx,1); lum=0.3*r+0.59*g+0.11*b
    p=max([f for f in F[str(n)] if f['kind']=='white'],key=lambda f:f['area']); x0,y0,x1,y1=p['bbox']
    disk=lambda q:(np.hypot(*np.mgrid[-q:q+1,-q:q+1])<=q)
    gold=(r-b>45)&(sat>0.28)
    m=(sat<0.13)&(lum<228)&~gold
    box=np.zeros_like(m); box[y0-8:y1+9,x0-8:x1+9]=True; m&=box
    m=ndimage.binary_closing(m,structure=disk(7)); m=ndimage.binary_fill_holes(m)
    m=ndimage.binary_opening(m,structure=disk(14))
    lab,k=ndimage.label(m); sizes=ndimage.sum(m,lab,range(1,k+1)); m=(lab==(1+int(np.argmax(sizes))))
    ys,xs=np.nonzero(m); pts=np.stack([xs,ys],1); hull=pts[ConvexHull(pts).vertices]
    bx0,bx1,by0,by1=xs.min(),xs.max(),ys.min(),ys.max()
    return dict(hull=hull.tolist(),body=[int(bx0),int(by0),int(bx1),int(by1)])
if __name__=='__main__':
    G={}; o=Image.new('RGB',(5*440,360),(255,255,255))
    for n in range(1,6):
        g=pear_geom(n); G[n]=g; x0,y0,x1,y1=g['body']; w=x1-x0; s=w/226.0
        tipy=min(g['hull'],key=lambda p:p[0])[1]
        beads=[(x0-13*s,tipy,19*s),(x1-28*s,y0+14*s,21*s),(x1-28*s,y1-12*s,21*s)]
        g['beads']=[[float(v) for v in b] for b in beads]
        cx=(x0+x1)//2; cy=(y0+y1)//2
        im=Image.open(f'std-{n}.png').convert('RGB'); d=ImageDraw.Draw(im)
        d.line([tuple(p) for p in g['hull']]+[tuple(g['hull'][0])],fill=(255,0,0),width=1)
        for bx,by,br in beads: d.ellipse([bx-br,by-br,bx+br,by+br],outline=(0,0,255))
        o.paste(im.crop((cx-220,cy-180,cx+220,cy+180)),((n-1)*440,0)); print(n,g['body'],'w',w,'h',y1-y0)
    json.dump(G,open('pears.json','w')); o.save('pears-outline.png')
