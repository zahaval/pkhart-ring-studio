import json, os, sys, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
src=open('rings3.py').read(); i=src.index("rounds=json.load(open('rounds.json'))")
ns={'__file__':os.path.abspath('rings3.py')}; exec(src[:i],ns); lift=ns['lift']
from marks import ROUNDS
if __name__=='__main__':
    R=json.load(open('../m2/rounds.json')); out={'1':R['1']}
    sheet=Image.new('RGB',(4*900,2*900),'white')
    for n,st in ROUNDS.items():
        std=np.asarray(Image.open(f'std-{n}.png').convert('RGB')).astype(float); bare=std.copy(); out[str(n)]=[]
        for kind,x,y,r in st:
            s=dict(kind=kind,x=x,y=y,r=r,wire_y=y); one,m,how=lift(std,s); print(n,kind,how)
            bare=bare*(1-m[...,None])+one*m[...,None]; out[str(n)].append(s)
        im=Image.fromarray(np.clip(bare,0,255).astype(np.uint8)); im.save(f'nostone-{n}.png')
        sheet.paste(Image.open(f'std-{n}.png').convert('RGB'),((n-2)*900,0)); sheet.paste(im,((n-2)*900,900))
    json.dump(out,open('rounds.json','w')); sheet.save('nostone-sheet.jpg',quality=88)
