"""Cut the 24 Nov gem sheet into the 12 Stone Bank stones: 300x300 px, stone radius 100 px, centred, transparent.
The sheet's background is a painted checkerboard, so each stone is cut by its shape: the stone's ellipse and its four
prong balls. Rose and silver copies differ only in the prongs."""
import os, sys, numpy as np
from PIL import Image
from scipy import ndimage
OUT='/Users/zahavalowy/Projects/pkhart-ring-studio/img/stones/'
NAMES=['garnet','amethyst','aquamarine','diamond','emerald','pearl','ruby','peridot','sapphire','opal','citrine','blue-topaz']
a=np.asarray(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'sheet.png')).convert('RGB')).astype(float)
H,W=a.shape[:2]; YY,XX=np.mgrid[:H,:W]
r,g,b=a[...,0],a[...,1],a[...,2]; mx=a.max(-1); mn=a.min(-1); sat=(mx-mn)/np.maximum(mx,1)
CX=[250.5,648,1030,1419.5]; CY=[186.5,465.5,766.5]; BX=74.3; BY=69.0
cells=[(CX[i%4],CY[i//4]) for i in range(12)]
# the stone's outline, measured on the coloured stones
fits=[]
for m,(cx,cy) in enumerate(cells):
    if NAMES[m] in ('diamond','pearl','opal','citrine'): continue
    near=(np.abs(XX-cx)<120)&(np.abs(YY-cy)<120)
    balls=np.zeros((H,W),bool)
    for sx in (-1,1):
        for sy in (-1,1): balls|=(XX-(cx+sx*BX))**2+(YY-(cy+sy*BY))**2<29**2
    st=ndimage.binary_opening((sat>0.30)&near&~balls,iterations=2); ys,xs=np.nonzero(st)
    rows=np.nonzero(st[:,int(cx)-6:int(cx)+7].any(1))[0]; cols=np.nonzero(st[int(cy)-6:int(cy)+7].any(0))[0]
    fits.append(((cols.min()+cols.max())/2-cx,(rows.min()+rows.max())/2-cy,(cols.max()-cols.min()+1)/2,(rows.max()-rows.min()+1)/2))
fits=np.array(fits); print(fits.round(1)); dx,dy,A,B=np.median(fits,0); print('centre offset',dx,dy,'semi-axes',A,B)
BR=27.0; BALLS=((72,68),(74,70),(72,70),(74,69))     # the prong balls, from the stone's centre
RAMP={'rose':[[0,[40,16,10]],[90,[138,70,56]],[160,[204,132,112]],[210,[236,180,160]],[240,[250,220,208]],[255,[255,244,240]]]}
def tint(rgb,metal):
    y=0.30*rgb[...,0]+0.59*rgb[...,1]+0.11*rgb[...,2]
    if metal=='silver': v=255*np.clip(y/255,0,1)**1.3; return np.stack([v*0.965,v*0.985,v],-1)
    st=RAMP['rose']; return np.stack([np.interp(y,[s[0] for s in st],[s[1][k] for s in st]) for k in range(3)],-1)
K=100.0/A; half=150/K
for metal in ('gold','rose','silver'):
    os.makedirs(OUT+metal,exist_ok=True)
    for m,(cx,cy) in enumerate(cells):
        x0=cx+dx-half; y0=cy+dy-half; box=(int(np.floor(x0))-2,int(np.floor(y0))-2,int(np.ceil(x0+2*half))+2,int(np.ceil(y0+2*half))+2)
        sub=a[box[1]:box[3],box[0]:box[2]].copy(); yy,xx=np.mgrid[box[1]:box[3],box[0]:box[2]]
        e=np.sqrt(((xx-cx-dx)/A)**2+((yy-cy-dy)/B)**2); stone=np.clip((1.0-e)*A/1.5+0.5,0,1)
        ball=np.zeros(stone.shape)
        fgs=(sat[box[1]:box[3],box[0]:box[2]]>0.14)|(a[box[1]:box[3],box[0]:box[2]].mean(-1)<205)
        fgs=ndimage.gaussian_filter(ndimage.binary_fill_holes(ndimage.binary_closing(fgs,iterations=4)).astype(float),0.8)
        for (sx,sy),(ox,oy) in zip(((-1,-1),(1,-1),(-1,1),(1,1)),BALLS):
            ball=np.maximum(ball,np.clip((BR-np.hypot(xx-(cx+dx+sx*ox),yy-(cy+dy+sy*oy)))/1.5+0.5,0,1))
        ball=ball*np.clip(fgs*1.6-0.3,0,1)
        if metal!='gold':
            mxs=sub.max(-1); mns=sub.min(-1); goldish=np.clip((sub[...,0]-sub[...,2]-25)/40.0,0,1) if NAMES[m]!='citrine' else np.ones(ball.shape)
            w=(ball*np.maximum(goldish,(ball>0.5)*(1-stone)))[...,None]; sub=sub*(1-w)+tint(sub,metal)*w
        al=np.maximum(stone,ball)
        im=Image.fromarray(np.dstack([np.clip(sub,0,255),al*255]).astype(np.uint8),'RGBA')
        im=im.transform((300,300),Image.AFFINE,(1/K,0,x0-box[0],0,1/K,y0-box[1]),resample=Image.BICUBIC)
        im.save(OUT+metal+f'/{m+1:02d}-{NAMES[m]}.png',optimize=True)
    Image.new('RGBA',(300,300),(0,0,0,0)).save(OUT+metal+'/seat.png')      # these stones carry their own setting
for f in ('06-alexandrite.png','10-pink-tourmaline.png','12-turquoise.png'):
    for metal in ('gold','rose','silver'):
        if os.path.exists(OUT+metal+'/'+f): os.remove(OUT+metal+'/'+f)
sh=Image.new('RGB',(12*150,3*150),(70,90,120))
for i,metal in enumerate(('gold','rose','silver')):
    for m in range(12):
        t=Image.open(OUT+metal+f'/{m+1:02d}-{NAMES[m]}.png').resize((150,150),Image.LANCZOS); sh.paste(t,(m*150,i*150),t)
sh.save('check.png')
