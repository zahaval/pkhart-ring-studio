"""Ring backgrounds for the configurator: each wrap count without its small stones (the pear stays),
in gold, rose and silver. Also the placement table: where each wrap's stone sits and how big."""
import json, os
import numpy as np
from PIL import Image
NEW='../new/'; OUT='/Users/zahavalowy/Projects/pkhart-ring-studio/img/rings/'
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
rounds=json.load(open(NEW+'rounds.json')); table={}
for n in range(1,6):
    a=np.asarray(Image.open(NEW+f'nostone-{n}.png').convert('RGB')).astype(float)
    for metal in ('gold','rose','silver'):
        px=a if metal=='gold' else tint(a,metal)
        px=clean_backdrop(px); os.makedirs(OUT+metal,exist_ok=True)
        Image.fromarray(np.clip(px,0,255).astype(np.uint8)).save(OUT+metal+f'/ring-{n}.jpg',quality=92)
    st=rounds[str(n)]; size=float(np.median([s['r'] for s in st if s['kind']!='white']))
    table[n]=dict(size=round(size,1),stones=[[s['x'],s['y']] for s in st])
json.dump(table,open('placements.json','w')); print(json.dumps(table))
# test composite: stones from the bank on the 5-wrap, default order and a shuffled one
N=['garnet','amethyst','aquamarine','diamond','emerald','alexandrite','ruby','peridot','sapphire','pink-tourmaline','citrine','turquoise']
def compose(n,metal,picks):
    base=Image.open(OUT+metal+f'/ring-{n}.jpg').convert('RGB'); t=table[n]
    for (x,y),m in zip(t['stones'],picks):
        if m is None: continue
        d=int(round(3*t['size'])); im=Image.open(f'/Users/zahavalowy/Projects/pkhart-ring-studio/img/stones/{metal}/{m+1:02d}-{N[m]}.png').resize((d,d),Image.LANCZOS)
        base.paste(im,(int(round(x-d/2)),int(round(y-d/2))),im)
    return base
sheet=Image.new('RGB',(3*600,600),(255,255,255))
sheet.paste(compose(5,'gold',[6,4,1,3,3]).resize((600,600),Image.LANCZOS),(0,0))
sheet.paste(compose(5,'silver',[0,8,9,10,11]).resize((600,600),Image.LANCZOS),(600,0))
sheet.paste(compose(3,'rose',[2,5,7]).resize((600,600),Image.LANCZOS),(1200,0))
sheet.save('compose-test.jpg',quality=92)
Image.open(NEW+'std-5.png').convert('RGB').crop((150,120,750,720)).save('orig5-crop.png'); compose(5,'gold',[6,4,1,3,3]).crop((150,120,750,720)).save('new5-crop.png')
