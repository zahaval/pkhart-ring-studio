"""Cut the Stone Bank sheet into 12 individual transparent stones, all the same size:
300x300 px, stone radius 100 px, centred. Rose and silver copies differ only in the four prongs."""
import json, os
import numpy as np
from PIL import Image
from scipy import ndimage
OUT='/Users/zahavalowy/Projects/pkhart-ring-studio/img/stones/'
NAMES=['garnet','amethyst','aquamarine','diamond','emerald','alexandrite','ruby','peridot','sapphire','pink-tourmaline','citrine','turquoise']
sheet=Image.open('../../images/34.webp').convert('RGB'); a=np.asarray(sheet).astype(float)
beads=json.load(open('beads.json'))
r_,g_,b_=a[:,:,0],a[:,:,1],a[:,:,2]; mx=a.max(2); mn=a.min(2); sat=(mx-mn)/np.maximum(mx,1)
H,W=sat.shape; YY,XX=np.mgrid[0:H,0:W]
goldm=(r_-b_>70)&(sat>0.45)&(r_>170)&(g_>110)&(g_<r_-15)
# rough stone centres: the middle of each stone's four prongs
pts=np.array([[b[0],b[1]] for b in beads]); used=np.zeros(len(pts),bool); rough=[]
for i in np.argsort(pts[:,1]*10+pts[:,0]):
    if used[i]: continue
    d=np.hypot(*(pts-pts[i]).T); idx=np.nonzero((d<140)&~used)[0]; used[idx]=True; rough.append(tuple(pts[idx].mean(0)))
rough=sorted(rough,key=lambda c:(round(c[1]/150),c[0])); assert len(rough)==12, len(rough)
def circle_fit(px,py):
    A=np.stack([px,py,np.ones_like(px)],1); bvec=-(px**2+py**2); sol=np.linalg.lstsq(A,bvec,rcond=None)[0]
    cx=-sol[0]/2; cy=-sol[1]/2; return cx,cy,np.sqrt(cx*cx+cy*cy-sol[2])
fits={}; beadrel={q:[] for q in range(4)}; beadr=[]
QUAD=[(-1,-1),(1,-1),(-1,1),(1,1)]
for m,(rx,ry) in enumerate(rough):
    if NAMES[m] in ('diamond','citrine'): continue
    near=np.hypot(XX-rx,YY-ry)<85
    stone=(sat>0.22)&near&~ndimage.binary_dilation(goldm,iterations=3)
    stone=ndimage.binary_opening(stone,iterations=1)
    edge=stone&~ndimage.binary_erosion(stone)
    ey,ex=np.nonzero(edge); dd=np.hypot(ex-rx,ey-ry); keep=dd>46
    # keep only the outline between the prongs (within 28 degrees of the four axes)
    ang=np.degrees(np.arctan2(ey-ry,ex-rx))%90; keep&=(ang<28)|(ang>62)
    cx,cy,R=circle_fit(ex[keep].astype(float),ey[keep].astype(float)); fits[m]=(cx,cy,R)
    for q,(sx,sy) in enumerate(QUAD):
        bm=goldm&near&((XX-cx)*sx>12)&((YY-cy)*sy>12)
        ys,xs=np.nonzero(bm)
        if len(ys)>120: beadrel[q].append((xs.mean()-cx,ys.mean()-cy)); beadr.append(np.sqrt(len(ys)/np.pi))
Rmed=float(np.median([f[2] for f in fits.values()])); off=np.median(np.array([[fits[m][0]-rough[m][0],fits[m][1]-rough[m][1]] for m in fits]),0)
brel=[np.median(np.array(beadrel[q]),0) for q in range(4)]; brad=float(np.median(beadr))
print('R per stone',{NAMES[m]:round(f[2],1) for m,f in fits.items()},'median',round(Rmed,2)); print('bead offsets',[b.round(1).tolist() for b in brel],'bead r',round(brad,1))
BEAD_R=12.9     # the prong beads' true radius on the sheet (the colour mask only catches their mid-tones)
stones=[]
for m,(rx,ry) in enumerate(rough):
    if m in fits: cx,cy,_=fits[m]
    elif NAMES[m]=='citrine':                       # its own gold body hides its prongs: take the grid position
        cx=(fits[2][0]+fits[6][0])/2.0; cy=(fits[8][1]+fits[9][1]+fits[11][1])/3.0
    else: cx,cy=rx+off[0],ry+off[1]
    bl=[]
    for q,(sx,sy) in enumerate(QUAD):
        bm=goldm&(np.hypot(XX-cx,YY-cy)<85)&((XX-cx)*sx>12)&((YY-cy)*sy>12); ys,xs=np.nonzero(bm)
        if NAMES[m]!='citrine' and len(ys)>120: bl.append((xs.mean(),ys.mean(),BEAD_R))
        else: bl.append((cx+brel[q][0],cy+brel[q][1],BEAD_R))
    stones.append((cx,cy,bl))
R=Rmed
RAMP={'rose':[[0,[52,22,14]],[0.30,[146,74,58]],[0.55,[201,129,108]],[0.80,[236,190,172]],[1,[255,241,234]]],
      'silver':[[0,[34,36,42]],[0.30,[104,109,118]],[0.55,[168,173,182]],[0.80,[226,230,236]],[1,[255,255,255]]]}
def tint(rgb,metal):
    """same metal recolour the ring uses: map the gold's light and dark onto the metal's own ramp"""
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]; mx=rgb.max(-1); mn=rgb.min(-1); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1),0)
    ok=(sat>=0.06)&(r>=b); amt=np.clip((sat-0.06)/0.18,0,1)*ok
    L=(0.30*r+0.59*g+0.11*b-40)/215; L=0.5+(L-0.5)*1.25; L=np.where(L>0.78,L+(L-0.78)*0.5,L); L=np.clip(L,0,1)
    stops=RAMP[metal]; ts=[s[0] for s in stops]; out=np.stack([np.interp(L,ts,[s[1][c] for s in stops]) for c in range(3)],-1)
    return rgb+(out-rgb)*amt[...,None]
S=300; RO=100.0; sc=RO/R
info=[]
for month,(cx,cy,bl) in enumerate(stones):
    half=S/2/sc
    box=(cx-half,cy-half,cx+half,cy+half)
    crop=sheet.resize((int(round(sheet.width*sc)),int(round(sheet.height*sc))),Image.LANCZOS).crop((int(round(cx*sc-S/2)),int(round(cy*sc-S/2)),int(round(cx*sc-S/2))+S,int(round(cy*sc-S/2))+S))
    ox=cx*sc-int(round(cx*sc-S/2)); oy=cy*sc-int(round(cy*sc-S/2))          # the stone's centre inside the crop
    rgb=np.asarray(crop).astype(float)
    yy,xx=np.mgrid[0:S,0:S]; dd=np.hypot(xx+0.5-ox,yy+0.5-oy)
    alpha=np.clip(RO-2.0-dd+0.5,0,1)
    bead=np.zeros((S,S))
    for bx_,by_,br_ in bl:
        bx=(bx_-cx)*sc+ox; by=(by_-cy)*sc+oy; br=br_*sc-0.5
        bead=np.maximum(bead,np.clip(br-np.hypot(xx+0.5-bx,yy+0.5-by)+0.5,0,1))
    alpha=np.maximum(alpha,bead)
    for metal in ('gold','rose','silver'):
        px=rgb.copy()
        if metal!='gold':
            t=tint(rgb,metal); w=np.clip(ndimage.grey_dilation(bead,size=3),0,1)[...,None]; px=rgb*(1-w)+t*w
        os.makedirs(OUT+metal,exist_ok=True)
        Image.fromarray(np.dstack([np.clip(px,0,255),alpha*255]).astype(np.uint8),'RGBA').save(OUT+metal+f'/{month+1:02d}-{NAMES[month]}.png',optimize=True)
    info.append(NAMES[month])
print(info)
chk=Image.new('RGB',(12*150,3*150),(120,120,120))
for mi,metal in enumerate(('gold','rose','silver')):
    for i,nm in enumerate(NAMES):
        im=Image.open(OUT+metal+f'/{i+1:02d}-{nm}.png').resize((150,150),Image.LANCZOS)
        bg=Image.new('RGB',(150,150),(255,255,255) if (i+mi)%2 else (40,40,40)); bg.paste(im,(0,0),im); chk.paste(bg,(i*150,mi*150))
chk.save('bank-check.png')
