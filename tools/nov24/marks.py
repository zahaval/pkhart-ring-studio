"""Hand marks for the 24 Nov photos, made on frames where each ring was 700px wide, and the conversion to the common
frame (every photo at the same scale, SC, so stones and bands keep the relative sizes of the supplied renders)."""
import json, os
D=os.path.dirname(os.path.abspath(__file__)); SC=1.08
_i=json.load(open(os.path.join(D,'info700.json'))); RATIO={n:1.0 for n in range(1,6)}
_ROUNDS={1:[('red',512,312,82)],
        2:[('red',447,302,67),('green',242,441,61.5)],
        3:[('red',450,241,60),('green',246,363,53),('purple',555,453,54)],
        4:[('red',454,235,58),('green',248,348,53.5),('purple',584,432,56),('white',313,522,50)],
        5:[('red',453,170,66),('green',248,304,53),('purple',577,414,57),('white',308,522,46),('white',624,604,40)]}
_PEAR={1:[345,430,668,672],2:[300,560,620,800],3:[285,520,598,745],4:[335,552,610,742],5:[280,615,585,830]}
_BEADS={1:[(322,558,33),(628,452,30),(630,650,32)], 2:[(274,686,33),(571,578,29),(571,778,31)], 3:[(278,625,33),(556,545,29),(550,731,31)],
        4:[(314,646,29),(564,569,28),(567,723,28),(588,648,34)], 5:[(258,722,30),(562,633,27),(548,808,30)]}
def P(n,x,y): k=RATIO[n]; return (450+(x-450)*k, 450+(y-450)*k)
ROUNDS={n:[(kind,)+tuple(round(v,1) for v in P(n,x,y))+(round(r*RATIO[n],1),) for kind,x,y,r in st] for n,st in _ROUNDS.items()}
PEAR={n:[int(round(v)) for v in P(n,b[0],b[1])+P(n,b[2],b[3])] for n,b in _PEAR.items()}
BEADS={n:[tuple(round(v,1) for v in P(n,x,y))+(round(r*RATIO[n]+1,1),) for x,y,r in bs] for n,bs in _BEADS.items()}
if __name__=='__main__':
    for n in range(1,6): print(n,round(RATIO[n],3),ROUNDS[n],PEAR[n])
