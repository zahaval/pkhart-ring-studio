"""Ring backgrounds for the configurator (no small stones, no bottom stone) in gold, rose and silver,
plus the placement table: each wrap's stone, its twin spot, and the bottom wire's tip."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
OUT='/Users/zahavalowy/Projects/pkhart-ring-studio/img/rings/'
RAMP={'rose':[[0,[52,22,14]],[0.30,[146,74,58]],[0.55,[201,129,108]],[0.80,[236,190,172]],[1,[255,241,234]]],
      'silver':[[0,[34,36,42]],[0.30,[104,109,118]],[0.55,[168,173,182]],[0.80,[226,230,236]],[1,[255,255,255]]]}
def tint(rgb,metal):
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]; mx=rgb.max(-1); mn=rgb.min(-1); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1),0)
    ok=(sat>=0.06)&(r>=b); amt=np.clip((sat-0.06)/0.18,0,1)*ok
    L=(0.30*r+0.59*g+0.11*b-40)/215; L=0.5+(L-0.5)*1.25; L=np.where(L>0.78,L+(L-0.78)*0.5,L); L=np.clip(L,0,1)
    stops=RAMP[metal]; ts=[s[0] for s in stops]; out=np.stack([np.interp(L,ts,[s[1][c] for s in stops]) for c in range(3)],-1)
    return rgb+(out-rgb)*amt[...,None]
def clean_backdrop(rgb):
    mn=rgb.min(-1); mx=rgb.max(-1); t=np.clip((mn-222)/24,0,1)*((mn>222)&(mx-mn<14))
    return rgb+(255-rgb)*t[...,None]
def slope_at(a,cx,cy,r):
    xL=int(round(cx-1.8*r)); xR=int(round(cx+1.8*r))
    def local(x0,x1):
        A=a[:,x0-1:x0+2].mean(1); B=a[:,x1-1:x1+2].mean(1); u0=int(cy-r); u1=int(cy+r); best=None
        for dd in range(-12,13):
            e=np.abs(A[u0:u1]-B[u0+dd:u1+dd]).mean()
            if best is None or e<best[0]: best=(e,dd)
        return best[1]/float(x1-x0)
    return (local(xL-28,xL)+local(xR,xR+28))/2.0
rounds=json.load(open('rounds.json')); bare=json.load(open('bare.json')); table={}
for n in range(1,6):
    a=np.asarray(Image.open(f'bare-{n}.png').convert('RGB')).astype(float)
    std=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float)
    for metal in ('gold','rose','silver'):
        px=clean_backdrop(a if metal=='gold' else tint(a,metal)); os.makedirs(OUT+metal,exist_ok=True)
        Image.fromarray(np.clip(px,0,255).astype(np.uint8)).save(OUT+metal+f'/ring-{n}.jpg',quality=92)
    st=rounds[str(n)]; size=float(np.median([s['r'] for s in st if s['kind']!='white']))
    stones=[]
    for s in st:
        sl=slope_at(std,s['x'],s['y'],s['r']); dx=(1 if s['x']<450 else -1)*2.25*size
        stones.append([s['x'],s['y'],round(s['x']+dx),round(s['y']+sl*dx)])
    table[n]=dict(size=round(size,1),stones=stones,tip=bare[str(n)]['tip'])
json.dump(table,open('rings-table.json','w'))
for n in range(1,6): print(f"  {n}:{{size:{table[n]['size']}, tip:{table[n]['tip']}, stones:{json.dumps(table[n]['stones']).replace(' ','')}}},")
