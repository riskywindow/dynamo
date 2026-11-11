"""compare.py REF.json OTHER.json [...]: greedy token identity + logprob deltas per prompt."""
import json, sys
ref = json.load(open(sys.argv[1]))
for path in sys.argv[2:]:
    o = json.load(open(path))
    print(f"== {ref['tag']}:{ref['gen']} vs {o['tag']}:{o['gen']}")
    allsame = True
    for a, b in zip(ref["results"], o["results"]):
        ta, tb = a["token_ids"], b["token_ids"]
        same = ta == tb
        allsame &= same
        div = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        n = min(len(ta), len(tb)) if div is None else div
        d = [abs(a["logprobs"][i]["lp"] - b["logprobs"][i]["lp"]) for i in range(n)
             if a["logprobs"][i]["lp"] is not None and b["logprobs"][i]["lp"] is not None]
        print(f"  {a['name']:8s} prompt={a['n_prompt']:6d} tokens {len(ta)}/{len(tb)} identical={same}"
              f" first_diff={div} max|dlogprob|={max(d) if d else 0:.6f} text={b['text'][:50]!r}")
    print(f"  ALL_IDENTICAL={allsame}")
