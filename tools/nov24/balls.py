"""Find the four prong balls of every round setting on the standardised photos (Hough circles near each corner of the
marked stone), so a setting can be lifted by its true outline."""
import json, os, sys, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw
from marks import ROUNDS
out={}; sheets=[]
for n in range(1,6):
    im=Image.open(f'std-{n}.png').convert('RGB'); a=np.asarray(im); gray=a.astype(float).mean(-1); d=ImageDraw.Draw(im); out[n]=[]
    from skimage.feature import canny
    from skimage.transform import hough_circle, hough_circle_peaks
    for kind,x,y,r in ROUNDS[n]:
        ox,oy=max(0,int(x-1.9*r)),max(0,int(y-1.9*r)); crop=gray[oy:int(y+1.9*r),ox:int(x+1.9*r)]
        ed=canny(crop,sigma=1.6,low_threshold=8,high_threshold=25); radii=np.arange(int(0.22*r),int(0.42*r)+1)
        hs=hough_circle(ed,radii); acc,cxs,cys,rs=hough_circle_peaks(hs,radii,min_xdistance=int(0.3*r),min_ydistance=int(0.3*r),total_num_peaks=40,normalize=True)
        cs=[(cx+ox,cy+oy,rr,ac) for ac,cx,cy,rr in zip(acc,cxs,cys,rs)]
        balls=[]
        for sx in (-1,1):
            for sy in (-1,1):
                ex,ey=x+sx*0.95*r,y+sy*0.84*r
                cand=[c for c in cs if np.hypot(c[0]-ex,c[1]-ey)<0.42*r and (c[0]-x)*sx>0.45*r and (c[1]-y)*sy>0.4*r]
                if cand: c=max(cand,key=lambda c:c[3]-0.004*np.hypot(c[0]-ex,c[1]-ey)); balls.append([float(c[0]),float(c[1]),float(c[2]),1])
                else: balls.append([ex,ey,0.32*r,0])
        out[n].append(balls)
        for bx,by,br,ok in balls: d.ellipse([bx-br,by-br,bx+br,by+br],outline=(0,255,0) if ok else (255,0,255),width=2)
        d.ellipse([x-0.87*r,y-0.87*r,x+0.87*r,y+0.87*r],outline=(0,180,255),width=1)
    sheets.append(im)
json.dump(out,open('balls.json','w'))
sh=Image.new('RGB',(4*700,560),'white')
for i,n in enumerate((2,3,4,5)): sh.paste(sheets[n-1].crop((100,100,800,660)),(i*700,0))
sh.save('balls.jpg',quality=90); print({n:[[b[3] for b in s] for s in out[n]] for n in out})
