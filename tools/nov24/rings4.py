"""Ring photos for the configurator, 24 Nov set (2 to 5 wraps; the 1-wrap photo is unchanged and is left alone).
Every stone on the page is a Stone Bank image, so each supplied photo has its own stones lifted off: only the footprint
of each setting is rebuilt from the band either side of it, the rest of the photo is untouched.
Writes, per metal: ring-N.jpg, and prints the placement table for index.html."""
import json, os, sys, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from scipy import ndimage
src=open('rings3.py').read(); ns={'__file__':os.path.abspath('rings3.py')}; exec(src[:src.index("def erase_curved")],ns)
tint,clean_backdrop,slope_at,changed=ns['tint'],ns['clean_backdrop'],ns['slope_at'],ns['changed']
from lift3 import bridge
OUT='/Users/zahavalowy/Projects/pkhart-ring-studio/img/rings/'
STONE_R=float(sys.argv[1]) if len(sys.argv)>1 else 48.0      # every birthstone, on every ring
BOTTOM=float(sys.argv[2]) if len(sys.argv)>2 else 2.0         # and one size of bottom stone (per carat) on every ring
P=json.load(open('params.json')); rounds=json.load(open('rounds.json')); bare_json=json.load(open('bare.json')); table={}
def ungem(a):
    """a stone's colour reflected in the band beside it would clash with whatever stone is chosen: turn it gold"""
    r,g,b=a[...,0],a[...,1],a[...,2]
    m=((r>90)&(g<75)&(b<90)&(r-g>45))|((g>r+25)&(g>b+25)&(g>60))|((b>g+35)&(r>g+15)&(b>90))|((r>150)&(g<110)&(r-g>80))
    m=ndimage.gaussian_filter(ndimage.binary_dilation(m,iterations=2).astype(float),1.5)[...,None]
    y=(0.45*r+0.45*g+0.10*b)+30; gold=np.stack([np.clip(y*1.2,0,255),np.clip(y*0.92,0,255),np.clip(y*0.42,0,255)],-1)
    return a*(1-m)+gold*m
def floor(a,top):
    """a plain white floor: the old stones' reflections and shadows under the ring go; the ring itself is untouched"""
    r,g,b=a[...,0],a[...,1],a[...,2]; mx=a.max(-1); mn=a.min(-1); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1),0)
    lab,k=ndimage.label((sat>0.26)&(r-b>45)); H,W=lab.shape; keep=np.zeros((H,W),bool)
    for i,sl in enumerate(ndimage.find_objects(lab)):
        if sl[0].start<top+60 and (lab[sl]==i+1).sum()>600: keep[sl]|=lab[sl]==i+1
    keep=ndimage.binary_dilation(ndimage.binary_closing(keep,iterations=3),iterations=2)
    yy=np.mgrid[:H,:W][0]; low=np.where(keep,yy,-1).max(0); ys=np.nonzero(keep.any(1))[0]; mid=(ys.min()+ys.max())//2
    low=np.where(low<0,mid,low); w=ndimage.gaussian_filter(((yy>low[None,:])&~keep).astype(float),0.8)
    return a+(255-a)*w[...,None]
for n in range(1,6):
    std=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float)
    bar=np.asarray(Image.open(f'bare-{n}.png').convert('RGB')).astype(float); st=rounds[str(n)]
    bare=std.copy(); gone=np.zeros(std.shape[:2])
    for k,s in enumerate(st):
        kw=dict(P.get(f'{n}-{k}',{})); [kw.pop(q,None) for q in ('x','y','r')]; one,m,_=bridge(std,s['x'],s['y'],s['r'],**kw)
        bare=bare*(1-m[...,None])+one*m[...,None]; gone=np.maximum(gone,m)
    nos=np.asarray(Image.open(f'nostone-{n}.png').convert('RGB')).astype(float)
    pm=changed(nos,bar); hard=ndimage.binary_dilation(np.abs(nos-bar).max(-1)>0,iterations=6)
    bare=bare*(1-pm[...,None])+np.where(hard[...,None],bar,bare)*pm[...,None]
    bare=floor(ungem(bare),bare_json[str(n)]['body'][1])
    for metal in ('gold','rose','silver'):
        px=clean_backdrop(bare if metal=='gold' else tint(bare,metal,None)); os.makedirs(OUT+metal,exist_ok=True)
        Image.fromarray(np.clip(px,0,255).astype(np.uint8)).save(OUT+metal+f'/ring-{n}.jpg',quality=93)
    stones=[]
    for s in st:
        sl=slope_at(std,s['x'],s['y'],s['r']); dx=(1 if s['x']<450 else -1)*2.25*STONE_R
        stones.append([round(s['x']),round(s['y']),round(s['x']+dx),round(s['y']+sl*dx)])
    B=bare_json[str(n)]; x0,y0,x1,y1=B['body']; h=y1-y0; w=x1-x0
    table[n]=dict(r=STONE_R,bottom=BOTTOM,seat=[B['right'],round(B['wire_at_right'])],top=y0,drop=round((y0+h/2-B['wire_at_right'])/h,2),stones=stones)
json.dump(table,open('rings-table.json','w'))
for n in range(1,6):
    t=table[n]; J=lambda v: json.dumps(v).replace(' ','')
    print(f"  {n}:{{r:{t['r']}, bottom:{t['bottom']}, seat:{J(t['seat'])}, top:{t['top']}, drop:{t['drop']},\n     stones:{J(t['stones'])}}},")
