"""Ring photos for the configurator. Every stone on the page is a Stone Bank image, so each supplied photo has its own
stones lifted off: only the footprint of each setting is replaced with bare wire, the rest of the photo is untouched.
Writes, per metal: ring-N.jpg, and prints the placement table for index.html."""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from scipy import ndimage
OUT='/Users/zahavalowy/Projects/pkhart-ring-studio/img/rings/'
RAMP={'rose':[[0,[52,22,14]],[0.30,[146,74,58]],[0.55,[201,129,108]],[0.80,[236,190,172]],[1,[255,241,234]]],
      'silver':[[0,[34,36,42]],[0.30,[104,109,118]],[0.55,[168,173,182]],[0.80,[226,230,236]],[1,[255,255,255]]]}
ROSE=[[0,[40,16,10]],[90,[138,70,56]],[160,[204,132,112]],[210,[236,180,160]],[240,[250,220,208]],[255,[255,244,240]]]
def tint(rgb,metal,keep=None):
    """gold -> rose / silver. Fully saturated gold takes the metal ramp; paler warm pixels (soft highlights, the gold
    reflected in the diamonds and on the floor) move to the same hue instead of keeping a yellow cast."""
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]; mx=rgb.max(-1); mn=rgb.min(-1); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1),0)
    warm=np.clip((r-b)/4.0,0,1); amt=np.clip((sat-0.06)/0.18,0,1)
    L=(0.30*r+0.59*g+0.11*b-40)/215; L=0.5+(L-0.5)*1.25; L=np.where(L>0.78,L+(L-0.78)*0.5,L); L=np.clip(L,0,1)
    stops=RAMP[metal]; ts=[s[0] for s in stops]; out=np.stack([np.interp(L,ts,[s[1][c] for s in stops]) for c in range(3)],-1)
    if metal=='silver':                                         # the photo's own light and shade, in a cool grey: the band keeps its body
        y=0.30*r+0.59*g+0.11*b; base=np.stack([y,y,y],-1); v=255*np.clip(y/255,0,1)**1.3
        out=np.stack([v*0.965,v*0.985,v],-1)
    else:                                                       # rose: the same, the photo's own light and shade in rose gold
        c=mx-mn; base=np.stack([mx,mx-0.77*c,mx-c],-1); y=0.30*r+0.59*g+0.11*b
        out=np.stack([np.interp(y,[s[0] for s in ROSE],[s[1][k] for s in ROSE]) for k in range(3)],-1)
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
rounds=json.load(open('rounds.json')); bare_json=json.load(open('bare.json')); table={}
STONE_R=43      # every Stone Bank stone is drawn at this radius, on every wrap of every ring
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
    # every stone comes from the Stone Bank, so the page's photo is the ring with its own stones lifted off: each one
    # replaced only inside its own footprint (the same small covers as before), the rest of the photo untouched
    bare=std.copy(); gone=np.zeros(std.shape[:2])
    for one,m in lifts: bare=bare*(1-m[...,None])+one*m[...,None]; gone=np.maximum(gone,m)
    pm=changed(nos,bar); hard=ndimage.binary_dilation(np.abs(nos-bar).max(-1)>0,iterations=6)
    bare=bare*(1-pm[...,None])+np.where(hard[...,None],bar,bare)*pm[...,None]; gone=np.maximum(gone,pm)
    for metal in ('gold','rose','silver'):
        fin=(lambda px,k=None: clean_backdrop(px if metal=='gold' else tint(px,metal,k))); os.makedirs(OUT+metal,exist_ok=True)
        Image.fromarray(np.clip(fin(bare,keep*(1-gone)),0,255).astype(np.uint8)).save(OUT+metal+f'/ring-{n}.jpg',quality=93)
    stones=[]
    for s in st:
        sl=slope_at(std,s['x'],s['y'],s['r']); dx=(1 if s['x']<450 else -1)*2.25*STONE_R
        stones.append([s['x'],s['y'],round(s['x']+dx),round(s['y']+sl*dx)])
    B=bare_json[str(n)]; x0,y0,x1,y1=B['body']; h=y1-y0; w=x1-x0
    table[n]=dict(bottom=round(w/152.3,2),seat=[B['right'],round(B['wire_at_right'])],top=y0,drop=round((y0+h/2-B['wire_at_right'])/h,2),stones=stones)
json.dump(table,open('rings-table.json','w'))
for n in range(1,6):
    t=table[n]; J=lambda v: json.dumps(v).replace(' ','')
    print(f"  {n}:{{bottom:{t['bottom']}, seat:{J(t['seat'])}, top:{t['top']}, drop:{t['drop']}, \n     stones:{J(t['stones'])}}},")
