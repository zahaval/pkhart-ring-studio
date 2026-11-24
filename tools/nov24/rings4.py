"""Ring photos for the configurator, 24 Nov set (2 to 5 wraps; the 1-wrap photo is unchanged and is left alone).
Every stone on the page is a Stone Bank image, so each supplied photo has its own stones lifted off: only the footprint
of each setting is rebuilt from the band either side of it, the rest of the photo is untouched.
Writes, per metal: ring-N.jpg, and prints the placement table for index.html."""
import json, os, sys, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from scipy import ndimage
src=open('rings3.py').read(); ns={'__file__':os.path.abspath('rings3.py')}; exec(src[:src.index("def erase_curved")],ns)
tint,clean_backdrop,slope_at,changed=ns['tint'],ns['clean_backdrop'],ns['slope_at'],ns['changed']
from lift2 import bridge
OUT='/Users/zahavalowy/Projects/pkhart-ring-studio/img/rings/'
STONE_R=float(sys.argv[1]) if len(sys.argv)>1 else 48.0
P=json.load(open('params.json')); rounds=json.load(open('rounds.json')); bare_json=json.load(open('bare.json')); table={}
def ungem(a):
    """a stone's colour reflected in the band beside it would clash with whatever stone is chosen: turn it gold"""
    r,g,b=a[...,0],a[...,1],a[...,2]
    m=((r>90)&(g<75)&(b<90)&(r-g>45))|((g>r+25)&(g>b+25)&(g>60))|((b>g+35)&(r>g+15)&(b>90))|((r>150)&(g<110)&(r-g>80))
    m=ndimage.gaussian_filter(ndimage.binary_dilation(m,iterations=2).astype(float),1.5)[...,None]
    y=(0.45*r+0.45*g+0.10*b)+30; gold=np.stack([np.clip(y*1.2,0,255),np.clip(y*0.92,0,255),np.clip(y*0.42,0,255)],-1)
    return a*(1-m)+gold*m
def floor(a,body):
    """the bottom stone's own reflection on the floor goes with it (the ring's soft reflection either side stays)"""
    x0,y0,x1,y1=body; H,W=a.shape[:2]; yy,xx=np.mgrid[:H,:W]
    mx=a.max(-1); mn=a.min(-1); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1),0); lum=0.3*a[...,0]+0.59*a[...,1]+0.11*a[...,2]
    pale=ndimage.gaussian_filter(((sat<0.30)&(lum>185)).astype(float),1.5)
    box=np.clip((xx-(x0-110))/90.0,0,1)*np.clip(((x1+110)-xx)/90.0,0,1)*np.clip((yy-(y0+y1)/2.0)/50.0,0,1)
    lab,k=ndimage.label(~((sat<0.30)&(lum>185))&(box>0.5))            # small leftovers of the old prongs on the floor
    if k:
        sz=ndimage.sum(np.ones_like(lab),lab,range(1,k+1)); speck=np.isin(lab,1+np.nonzero(sz<400)[0])
        pale=np.maximum(pale,ndimage.gaussian_filter(ndimage.binary_dilation(speck,iterations=3).astype(float),1.2))
    w=(pale*box)[...,None]; return a+(255-a)*w
for n in range(2,6):
    std=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float)
    bar=np.asarray(Image.open(f'bare-{n}.png').convert('RGB')).astype(float); st=rounds[str(n)]
    bare=std.copy(); gone=np.zeros(std.shape[:2])
    for k,s in enumerate(st):
        one,m,_=bridge(std,s['x'],s['y'],s['r'],**{a:b for a,b in P.get(f'{n}-{k}',{}).items() if a in('padL','padR','hy','feather','mix','body','prong','cap')})
        bare=bare*(1-m[...,None])+one*m[...,None]; gone=np.maximum(gone,m)
    nos=np.asarray(Image.open(f'nostone-{n}.png').convert('RGB')).astype(float)
    pm=changed(nos,bar); hard=ndimage.binary_dilation(np.abs(nos-bar).max(-1)>0,iterations=6)
    bare=bare*(1-pm[...,None])+np.where(hard[...,None],bar,bare)*pm[...,None]
    bare=floor(ungem(bare),bare_json[str(n)]['body'])
    for metal in ('gold','rose','silver'):
        px=clean_backdrop(bare if metal=='gold' else tint(bare,metal,None)); os.makedirs(OUT+metal,exist_ok=True)
        Image.fromarray(np.clip(px,0,255).astype(np.uint8)).save(OUT+metal+f'/ring-{n}.jpg',quality=93)
    stones=[]
    for s in st:
        sl=slope_at(std,s['x'],s['y'],s['r']); dx=(1 if s['x']<450 else -1)*2.25*STONE_R
        stones.append([round(s['x']),round(s['y']),round(s['x']+dx),round(s['y']+sl*dx)])
    B=bare_json[str(n)]; x0,y0,x1,y1=B['body']; h=y1-y0; w=x1-x0
    table[n]=dict(r=STONE_R,bottom=round(w/152.3,2),seat=[B['right'],round(B['wire_at_right'])],top=y0,drop=round((y0+h/2-B['wire_at_right'])/h,2),stones=stones)
json.dump(table,open('rings-table.json','w'))
for n in range(2,6):
    t=table[n]; J=lambda v: json.dumps(v).replace(' ','')
    print(f"  {n}:{{r:{t['r']}, bottom:{t['bottom']}, seat:{J(t['seat'])}, top:{t['top']}, drop:{t['drop']},\n     stones:{J(t['stones'])}}},")
