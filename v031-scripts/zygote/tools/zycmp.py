import json, sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
for x,y in zip(a,b):
    d=next((i for i,(s,t) in enumerate(zip(x["tok"],y["tok"])) if s!=t), None); n=len(x["tok"]) if d is None else d
    dl=max([abs(p-q) for p,q in zip(x["lp"][:n],y["lp"][:n]) if p is not None and q is not None] or [0])
    print(f"identical={x['tok']==y['tok']} first_diff={d} max_dlogprob={dl:.6f} {x['text'][:40]!r}")
