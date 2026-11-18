"""Cut the bottom-stone sheet (5 cuts x 3 sizes) into 15 individual transparent stones.
Every file keeps the sheet's own scale (3 px per sheet px), so the sizes stay true to each other;
each stone is centred in its file. Rose and silver copies differ only in the prongs."""
import json, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull
OUT='/Users/zahavalowy/Projects/pkhart-ring-studio/img/bottom/'
K=3
sheet=Image.open('../../images/35.webp').convert('RGB')
big=sheet.resize((sheet.width*K,sheet.height*K),Image.LANCZOS); a=np.asarray(big).astype(float); H,W,_=a.shape
r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]; mx=a.max(2); mn=a.min(2); sat=(mx-mn)/np.maximum(mx,1); lum=0.3*r+0.59*g+0.11*b
disk=lambda q:(np.hypot(*np.mgrid[-q:q+1,-q:q+1])<=q)
gold=(r-b>60)&(sat>0.40)&(r>150)&(g>95)&(g<r-10); gold=ndimage.binary_opening(gold,iterations=2)
marks=(lum<222)&~gold                                   # anything darker than the checkerboard
body=ndimage.binary_closing(marks,structure=disk(7*K//3*2)); body=ndimage.binary_fill_holes(body)
body=ndimage.binary_opening(body,structure=disk(9))
lab,k=ndimage.label(body); stones=[]
for i in range(1,k+1):
    ys,xs=np.nonzero(lab==i)
    if ys.max()-ys.min()>40*K and xs.max()-xs.min()>40*K: stones.append(i)
assert len(stones)==15, len(stones)
cent=[ndimage.center_of_mass(lab==i) for i in stones]
rows=sorted(range(15),key=lambda j:cent[j][0]); grid=[sorted(rows[q*5:q*5+5],key=lambda j:cent[j][1]) for q in range(3)]
CUTS=['round','emerald','heart','oval','pear']; SIZES=['030','070','100']
RAMP={'rose':[[0,[52,22,14]],[0.30,[146,74,58]],[0.55,[201,129,108]],[0.80,[236,190,172]],[1,[255,241,234]]],
      'silver':[[0,[34,36,42]],[0.30,[104,109,118]],[0.55,[168,173,182]],[0.80,[226,230,236]],[1,[255,255,255]]]}
def tint(rgb,metal):
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]; mx=rgb.max(-1); mn=rgb.min(-1); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1),0)
    ok=(sat>=0.06)&(r>=b); amt=np.clip((sat-0.06)/0.18,0,1)*ok
    L=(0.30*r+0.59*g+0.11*b-40)/215; L=0.5+(L-0.5)*1.25; L=np.where(L>0.78,L+(L-0.78)*0.5,L); L=np.clip(L,0,1)
    stops=RAMP[metal]; ts=[s[0] for s in stops]; out=np.stack([np.interp(L,ts,[s[1][c] for s in stops]) for c in range(3)],-1)
    return rgb+(out-rgb)*amt[...,None]
def hull_mask(m):
    ys,xs=np.nonzero(m); pts=np.stack([xs,ys],1); hp=pts[ConvexHull(pts).vertices]
    im=Image.new('L',(m.shape[1],m.shape[0]),0); ImageDraw.Draw(im).polygon([tuple(p) for p in hp.tolist()],fill=255); return np.asarray(im)>0
table={}; chk=Image.new('RGB',(5*420,3*560),(120,120,120))
for ri,row in enumerate(grid):
    for ci,j in enumerate(row):
        m=lab==stones[j]; cut=CUTS[ci]
        zone=ndimage.binary_dilation(m,structure=disk(14))
        clean=m&~ndimage.binary_dilation(gold&zone,structure=disk(9))          # the stone without its prongs
        clean=ndimage.binary_opening(clean,structure=disk(5))
        cl,ck=ndimage.label(clean); clean=cl==(1+int(np.argmax(ndimage.sum(clean,cl,range(1,ck+1)))))
        shape=hull_mask(clean)
        if cut=='heart':
            ys,xs=np.nonzero(shape); ax=(xs.min()+xs.max())/2.0; top=ys.min(); hgt=ys.max()-ys.min(); wid=xs.max()-xs.min()
            YY0,XX0=np.mgrid[0:H,0:W]
            cleft=shape&~m&(np.abs(XX0-ax)<0.30*wid)&(YY0<top+0.35*hgt)
            cleft=cleft|cleft[:,::-1] if False else cleft
            # mirror the cleft about the heart's own axis so both lobes match
            x0=int(round(ax-0.32*wid)); x1=int(round(ax+0.32*wid)); sub=cleft[:,x0:x1+1]; cleft[:,x0:x1+1]=sub|sub[:,::-1]
            cleft=ndimage.binary_opening(cleft,structure=disk(3))
            shape=shape&~cleft
        shape=ndimage.binary_erosion(shape,structure=disk(2))                  # trim the soft halo round the edge
        ys,xs=np.nonzero(shape); bx0,bx1,by0,by1=xs.min(),xs.max(),ys.min(),ys.max(); cx=(bx0+bx1)/2.0; cy=(by0+by1)/2.0
        # the prongs that belong to this stone
        gl,gk=ndimage.label(ndimage.binary_closing(gold&zone,structure=disk(3))); bead=np.zeros_like(shape,float)
        YY,XX=np.mgrid[0:H,0:W]
        for q in range(1,gk+1):
            py,px=np.nonzero(gl==q)
            if len(py)<150: continue
            br=np.sqrt(len(py)/np.pi)*1.22+1.0
            bead=np.maximum(bead,np.clip(br-np.hypot(XX-px.mean(),YY-py.mean())+0.5,0,1))
        alpha=np.maximum(ndimage.gaussian_filter(shape.astype(float),1.3),bead)
        ays,axs=np.nonzero(alpha>0.02); hw=int(np.ceil(max(cx-axs.min(),axs.max()-cx)))+3; hh=int(np.ceil(max(cy-ays.min(),ays.max()-cy)))+3
        X0=int(round(cx))-hw; Y0=int(round(cy))-hh; box=(slice(Y0,Y0+2*hh),slice(X0,X0+2*hw))
        rgb=a[box]; al=alpha[box]; bd=np.clip(ndimage.grey_dilation(bead[box],size=3),0,1)[...,None]
        name=f'{cut}-{SIZES[ri]}.png'
        for metal in ('gold','rose','silver'):
            px_=rgb if metal=='gold' else rgb*(1-bd)+tint(rgb,metal)*bd
            os.makedirs(OUT+metal,exist_ok=True)
            Image.fromarray(np.dstack([np.clip(px_,0,255),al*255]).astype(np.uint8),'RGBA').save(OUT+metal+'/'+name,optimize=True)
        table[f'{cut}-{SIZES[ri]}']=dict(w=round(2*hw/K,1),h=round(2*hh/K,1),bw=round((bx1-bx0)/K,1),bh=round((by1-by0)/K,1))
        im=Image.open(OUT+'gold/'+name); bgc=Image.new('RGB',(420,560),(30,30,30) if (ri+ci)%2 else (255,255,255)); bgc.paste(im,((420-im.width)//2,(560-im.height)//2),im); chk.paste(bgc,(ci*420,ri*560))
json.dump(table,open('bottom-table.json','w')); print(json.dumps(table))
chk.resize((1050,840),Image.LANCZOS).save('bottom-check.png')
