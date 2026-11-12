#!/usr/bin/env python3
"""promote_stats.py ARM_DIR: kill -> serving and post-promotion latency for a promote031.sh arm."""
import json, os, sys

d = sys.argv[1]
reqs = [json.loads(l) for l in open(os.path.join(d, "requests.jsonl")) if l.strip()]
ev = {}
for l in open(os.path.join(d, "events.txt")):
    k, t = l.split()
    ev.setdefault(k, float(t))
kill = ev["kill"]


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else None


inflight = [r for r in reqs if r["t_send"] < kill < r.get("t_end", 0)]
after = [r for r in reqs if r["t_send"] >= kill]
ok_after = [r for r in after if not r.get("error") and "t_first" in r]
first_tok = min((r["t_first"] for r in ok_after), default=None)
post = [r for r in ok_after if r["t_send"] >= ev.get("serving", kill)]
ttft = [r["t_first"] - r["t_send"] for r in post]
itl = []
for r in post:
    t = [r["t_first"] + c for c in r.get("chunk_t", [])]
    itl += [b - a for a, b in zip(t, t[1:])]
print(f"events (s after kill): " + ", ".join(f"{k}={v - kill:+.2f}" for k, v in sorted(ev.items(), key=lambda x: x[1])))
print(f"in flight at kill: {len(inflight)} (errors {sum(1 for r in inflight if r.get('error'))})")
print(f"sent after kill: {len(after)}, errors {sum(1 for r in after if r.get('error'))}")
print(f"kill -> first token of a request sent after kill: {first_tok - kill:.2f} s" if first_tok else "no token after kill")
print(f"post-promotion (sent after serving): n={len(post)} TTFT p50/p90 {pct(ttft, 50):.2f}/{pct(ttft, 90):.2f} s, "
      f"ITL p50/p90/p99 {pct(itl, 50) * 1000:.0f}/{pct(itl, 90) * 1000:.0f}/{pct(itl, 99) * 1000:.0f} ms" if post else "no post-promotion requests")
