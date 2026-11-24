import json, os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift_all import ROUNDS
P={}
for n,st in ROUNDS.items():
    for k,(kind,x,y,r) in enumerate(st):
        if kind=='red': P[f'{n}-{k}']=dict(padL=1.5,padR=1.5)
        elif x<450: P[f'{n}-{k}']=dict(padL=1.45,padR=1.9)
        else: P[f'{n}-{k}']=dict(padL=1.9,padR=1.45)
json.dump(P,open('params.json','w'),indent=0)
