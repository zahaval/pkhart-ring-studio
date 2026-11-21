"""Ring photos for the configurator. The page shows each supplied photo as it is. A stone the customer changes is covered
by a small "lift" sprite (the bare wire under that one stone) and the Stone Bank stone goes on top; the same for the pear.
Writes, per metal: photo-N.jpg, lift-N-<wrap>.png, lift-N-b.png, and prints the placement table for index.html."""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from scipy import ndimage
OUT='/Users/zahavalowy/Projects/pkhart-ring-studio/img/rings/'
RAMP={'rose':[[0,[52,22,14]],[0.30,[146,74,58]],[0.55,[201,129,108]],[0.80,[236,190,172]],[1,[255,241,234]]],
      'silver':[[0,[34,36,42]],[0.30,[104,109,118]],[0.55,[168,173,182]],[0.80,[226,230,236]],[1,[255,255,255]]]}
def tint(rgb,metal,keep=None):
    """gold -> rose / silver. Fully saturated gold takes the metal ramp; paler warm pixels (soft highlights, the gold
    reflected in the diamonds and on the floor) move to the same hue instead of keeping a yellow cast."""
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]; mx=rgb.max(-1); mn=rgb.min(-1); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1),0)
    warm=np.clip((r-b)/4.0,0,1); amt=np.clip((sat-0.06)/0.18,0,1)
    L=(0.30*r+0.59*g+0.11*b-40)/215; L=0.5+(L-0.5)*1.25; L=np.where(L>0.78,L+(L-0.78)*0.5,L); L=np.clip(L,0,1)
    stops=RAMP[metal]; ts=[s[0] for s in stops]; out=np.stack([np.interp(L,ts,[s[1][c] for s in stops]) for c in range(3)],-1)
    if metal=='silver':
        y=0.30*r+0.59*g+0.11*b; base=np.stack([y,y,y],-1)
    else:
        c=mx-mn; base=np.stack([mx,mx-0.77*c,mx-c],-1)
    mix=base+(out-base)*amt[...,None]; w=warm
    if keep is not None: w=w*(1-keep)                           # the photo's own coloured stones keep their colour
    return rgb+(mix-rgb)*w[...,None]
def clean_backdrop(rgb):
    mn=rgb.min(-1); mx=rgb.max(-1); t=np.clip((mn-222)/24,0,1)*((mn>222)&(mx-mn<14))
    return rgb+(255-rgb)*t[...,None]
def stone_colour(a,stones):
    a=a.astype(int); r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
    m=((r>90)&(g<75)&(b<90)&(r-g>45))|((g>r+25)&(g>b+25)&(g>60))|((b>g+35)&(r>g+15)&(b>90))
    m=ndimage.binary_closing(ndimage.binary_opening(m,iterations=1),iterations=4)
    yy,xx=np.mgrid[:a.shape[0],:a.shape[1]]; near=np.zeros(m.shape,bool)
    for s in stones: near|=(xx-s['x'])**2+(yy-s['y'])**2<(1.15*s['r'])**2
    return ndimage.gaussian_filter(ndimage.binary_dilation(m&near,iterations=2).astype(float),1.0)
def slope_at(a,cx,cy,r):
    xL=int(round(cx-1.8*r)); xR=int(round(cx+1.8*r))
    def local(x0,x1):
        A=a[:,x0-1:x0+2].mean(1); B=a[:,x1-1:x1+2].mean(1); u0=int(cy-r); u1=int(cy+r); best=None
        for dd in range(-12,13):
            e=np.abs(A[u0:u1]-B[u0+dd:u1+dd]).mean()
            if best is None or e<best[0]: best=(e,dd)
        return best[1]/float(x1-x0)
    return (local(xL-28,xL)+local(xR,xR+28))/2.0
def changed(a,b,near=None,thr=3):
    """soft mask of where b differs from a (optionally only inside a box)"""
    m=np.abs(a-b).max(-1)>thr
    if near is not None:
        x0,y0,x1,y1=near; k=np.zeros_like(m); k[max(0,y0):y1,max(0,x0):x1]=True; m&=k
    m=ndimage.binary_dilation(ndimage.binary_closing(m,iterations=2),iterations=3)
    return ndimage.gaussian_filter(m.astype(float),1.4)
def sprite(path,rgb,alpha):
    ys,xs=np.nonzero(alpha>0.004); x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    px=np.dstack([np.clip(rgb[y0:y1,x0:x1],0,255),np.clip(alpha[y0:y1,x0:x1]*255,0,255)]).astype(np.uint8)
    Image.fromarray(px,'RGBA').save(path,optimize=True); return [int(x0),int(y0),int(x1-x0),int(y1-y0)]
rounds=json.load(open('rounds.json')); bare=json.load(open('bare.json')); table={}
for f in glob.glob(OUT+'*/ring-*.jpg')+glob.glob(OUT+'*/lift-*.png')+glob.glob(OUT+'*/photo-*.jpg'): os.remove(f)
from erase import erase_round
for n in range(1,6):
    std=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float)
    nos=np.asarray(Image.open(f'nostone-{n}.png').convert('RGB')).astype(float)
    bar=np.asarray(Image.open(f'bare-{n}.png').convert('RGB')).astype(float)
    st=rounds[str(n)]; keep=stone_colour(std,st)
    # one sprite per stone: the photo with only that stone lifted
    lifts=[]; masks=[]
    for s in st:
        one,(xL,xR,cL,cR)=erase_round(std.copy(),s['x'],s['y'],s['r'])
        m=changed(std,one,(xL-6,int(min(cL,cR)-1.5*s['r'])-10,xR+7,int(max(cL,cR)+1.5*s['r'])+11),thr=22); lifts.append((one,m)); masks.append(m)
    pm=changed(nos,bar); pear_rgb=std.copy(); hard=(np.abs(nos-bar).max(-1)>0); pear_rgb=np.where(ndimage.binary_dilation(hard,iterations=6)[...,None],bar,std)
    over=max(float((pm*m).max()) for m in masks); print(n,'pear/stone sprite overlap',round(over,3))
    boxes=[]
    for metal in ('gold','rose','silver'):
        fin=(lambda px,k=None: clean_backdrop(px if metal=='gold' else tint(px,metal,k))); os.makedirs(OUT+metal,exist_ok=True)
        Image.fromarray(np.clip(fin(std,keep),0,255).astype(np.uint8)).save(OUT+metal+f'/photo-{n}.jpg',quality=93)
        boxes=[sprite(OUT+metal+f'/lift-{n}-{i+1}.png',fin(one,keep*(1-m)),m) for i,(one,m) in enumerate(lifts)]
        pbox=sprite(OUT+metal+f'/lift-{n}-b.png',fin(pear_rgb,keep),pm)
    stones=[]
    for s,bx in zip(st,boxes):
        sl=slope_at(std,s['x'],s['y'],s['r']); dx=(1 if s['x']<450 else -1)*2.25*s['r']
        stones.append([s['x'],s['y'],round(s['x']+dx),round(s['y']+sl*dx),round(s['r'],1)]+bx)
    B=bare[str(n)]; x0,y0,x1,y1=B['body']; h=y1-y0; w=x1-x0
    table[n]=dict(bottom=round(w/152.3,2),seat=[B['right'],round(B['wire_at_right'])],top=y0,drop=round((y0+h/2-B['wire_at_right'])/h,2),pear=[x0,y0,w,h],lift=pbox,stones=stones)
json.dump(table,open('rings-table.json','w'))
for n in range(1,6):
    t=table[n]; J=lambda v: json.dumps(v).replace(' ','')
    print(f"  {n}:{{bottom:{t['bottom']}, seat:{J(t['seat'])}, top:{t['top']}, drop:{t['drop']}, pear:{J(t['pear'])}, lift:{J(t['lift'])},\n     stones:{J(t['stones'])}}},")
