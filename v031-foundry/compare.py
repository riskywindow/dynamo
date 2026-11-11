#!/usr/bin/env python3
"""Compare probe.py outputs: tokens identical? max |logprob delta| per set. First file = reference.
Usage: compare.py ref.json other.json [...]"""
import json, sys


def main():
    ref = json.load(open(sys.argv[1]))
    for p in sys.argv[2:]:
        o = json.load(open(p))
        print("%s vs %s:" % (o["label"], ref["label"]))
        for s, rows in ref["sets"].items():
            same = n = 0; dmax = 0.0; first_div = []
            for i, (a, b) in enumerate(zip(rows, o["sets"].get(s, []))):
                if "error" in a or "error" in b: continue
                n += 1
                k = next((j for j in range(min(len(a["tokens"]), len(b["tokens"]))) if a["tokens"][j] != b["tokens"][j]), None)
                if k is None and len(a["tokens"]) == len(b["tokens"]): same += 1
                else: first_div.append((i, k))
                m = min(len(a["lp"]), len(b["lp"]), k if k is not None else 10**9)
                d = [abs(x - y) for x, y in zip(a["lp"][:m], b["lp"][:m]) if x is not None and y is not None]
                dmax = max([dmax] + d)
            print("  %-7s identical %d/%d  max|dlp| (matched prefix) %.4f  diverge(idx,pos) %s" % (s, same, n, dmax, first_div[:4]))


if __name__ == "__main__":
    main()
