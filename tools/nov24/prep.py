"""Standardise the five supplied ring images to the builder frame (900px, ring 700px wide, centred)
and find every stone: coloured rounds by hue, diamonds as grey blobs, the pear as the big one."""
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
SRC='gen/'
FILES={1:'1.png',2:'2.png',3:'3.png',4:'4.png',5:'5.png'}
S=4  # work at 4x of the 900 frame? no: keep native, record the transform
info={}
sheet=Image.new('RGB',(5*450,450),(255,255,255))
for n,f in FILES.items():
    im=Image.open(SRC+f).convert('RGB'); a=np.asarray(im).astype(int); H,W,_=a.shape
    r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]; mx=a.max(2); mn=a.min(2); sat=(mx-mn)/np.maximum(mx,1)
    gold=(r-b>60)&(sat>0.35)&(r>150)&(g>100)&(g<r)
    gold=ndimage.binary_opening(gold,iterations=2)
    ys,xs=np.nonzero(gold); x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
    # everything that is not white backdrop (ring, stones, pear): used for the vertical centre
    obj=(mn<232)|(sat>0.2); obj=ndimage.binary_opening(obj,iterations=3)
    lab,k=ndimage.label(obj); sizes=ndimage.sum(obj,lab,range(1,k+1)); big=lab==(1+int(np.argmax(sizes)))
    oy,ox=np.nonzero(big); by0,by1=oy.min(),oy.max()
    sc=700.0/(x1-x0); cx=(x0+x1)/2; cy=(by0+by1)/2
    side=900/sc; L=cx-side/2; T=cy-side/2
    crop=im.crop((int(round(L)),int(round(T)),int(round(L+side)),int(round(T+side))))   # PIL pads outside with black
    canvas=Image.new('RGB',(int(round(side)),int(round(side))),(255,255,255))
    canvas.paste(im,(-int(round(L)),-int(round(T))))
    std=canvas.resize((900,900),Image.LANCZOS); std.save(f'std-{n}.png')
    info[n]=dict(scale=sc,L=L,T=T,ring=[int(x0),int(x1),int(y0),int(y1)],obj=[int(by0),int(by1)],size=[W,H])
    sheet.paste(std.resize((450,450),Image.LANCZOS),((n-1)*450,0))
    print(n,f,'ring w',x1-x0,'scale',round(sc,3),'L,T',round(L),round(T),'side',round(side))
json.dump(info,open('info.json','w')); sheet.save('std-sheet.jpg',quality=88)
