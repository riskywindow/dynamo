#!/usr/bin/env python3
"""sbs_analyze.py OUT.json ARM=DIR[,DIR...] ...: side-by-side replenishment metrics, one window definition for all arms.

Each DIR is one cycle: requests.jsonl (per-chunk loadgen) + loadgen_start.json, and either events.txt (GMS chain:
kill, promoted, rep_start, rep_parked) or events.json (Snapshot: t_kill, t_promoted, t_rep_start, t_rep_ready).
Windows (absolute times):
  pre        [t0+30, kill)                    (baseline: [t0+30, t0+120))
  failover   [kill, promoted)
  replenish  [rep_start, rep_parked]          (baseline: the mean offsets from t0 of the arm given by --ref)
  post       (rep_parked, t0+duration]
Per window: TTFT p50/p90 (requests sent in it), ITL p50/p99 (inter-chunk gaps ending in it), stalls = inter-chunk
gaps > 0.5 s in any stream, merged across streams, counted if they start in the window.
Per cycle: replenish_s = rep_parked - rep_start; kill->first token = first token of a request sent after the kill;
failed/truncated requests.
"""
import json, os, statistics, sys

STALL = 0.5


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def load(d):
    reqs = [json.loads(l) for l in open(os.path.join(d, "requests.jsonl")) if l.strip()]
    st = json.load(open(os.path.join(d, "loadgen_start.json")))
    ev = {}
    if os.path.exists(os.path.join(d, "events.json")):
        e = json.load(open(os.path.join(d, "events.json")))
        ev = {"kill": e.get("t_kill"), "promoted": e.get("t_promoted"), "rep_start": e.get("t_rep_start"),
              "rep_parked": e.get("t_rep_ready")}
    elif os.path.exists(os.path.join(d, "events.txt")):
        for l in open(os.path.join(d, "events.txt")):
            k, t = l.split()
            ev.setdefault(k, float(t))
    return reqs, st["t0"], st.get("duration", 420.0), {k: v for k, v in ev.items() if v}


def gaps_of(reqs):
    gaps = []
    for r in reqs:
        if r.get("kind") != "measured" or "t_first" not in r:
            continue
        t = [r["t_first"] + c for c in r.get("chunk_t", [])]
        gaps += list(zip(t, t[1:]))
    merged = []
    for a, b in sorted((a, b) for a, b in gaps if b - a > STALL):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return gaps, merged


def window(reqs, gaps, merged, a, b):
    ttft = [r["t_first"] - r["t_send"] for r in reqs
            if r.get("kind") == "measured" and "t_first" in r and a <= r["t_send"] < b]
    itl = [y - x for x, y in gaps if a <= y < b]
    st = [(s, e) for s, e in merged if a <= s < b]
    return {"s": round(b - a, 1), "n_req": len(ttft), "ttft_p50": pct(ttft, 50), "ttft_p90": pct(ttft, 90),
            "itl_p50_ms": pct(itl, 50) and pct(itl, 50) * 1e3, "itl_p99_ms": pct(itl, 99) and pct(itl, 99) * 1e3,
            "stalls": len(st), "stall_s": round(sum(e - s for s, e in st), 2),
            "max_stall_s": round(max((e - s for s, e in st), default=0), 2)}


def cycle(d, ref_offsets=None):
    reqs, t0, dur, ev = load(d)
    gaps, merged = gaps_of(reqs)
    out = {"dir": d, "events": {k: round(v - t0, 2) for k, v in ev.items()}}
    measured = [r for r in reqs if r.get("kind") == "measured"]
    out["n_measured"] = len(measured)
    out["errors"] = sum(1 for r in measured if "error" in r)
    out["truncated"] = sum(1 for r in measured if "error" not in r and "t_first" in r and len(r.get("chunk_t", [])) < 1000)
    if "kill" in ev:
        k = ev["kill"]
        out["windows"] = {"pre": window(reqs, gaps, merged, t0 + 30, k)}
        if "promoted" in ev:
            out["windows"]["failover"] = window(reqs, gaps, merged, k, ev["promoted"])
            out["kill_to_promoted_s"] = round(ev["promoted"] - k, 2)
        after = [r["t_first"] for r in measured if "t_first" in r and r["t_send"] >= k]
        out["kill_to_first_token_s"] = round(min(after) - k, 2) if after else None
        if "rep_start" in ev and "rep_parked" in ev:
            out["replenish_s"] = round(ev["rep_parked"] - ev["rep_start"], 2)
            out["windows"]["replenish"] = window(reqs, gaps, merged, ev["rep_start"], ev["rep_parked"])
            out["windows"]["post"] = window(reqs, gaps, merged, ev["rep_parked"], t0 + dur)
        if "pre_start" in ev and "pre_held" in ev:
            # Gated spare (chain031g.sh): its startup up to the gate runs while the promoted engine serves.
            out["prestart_s"] = round(ev["pre_held"] - ev["pre_start"], 2)
            out["windows"]["prestart"] = window(reqs, gaps, merged, ev["pre_start"], ev["pre_held"])
    else:
        out["windows"] = {"pre": window(reqs, gaps, merged, t0 + 30, t0 + 120)}
        if ref_offsets:
            a, b = ref_offsets
            out["windows"]["replenish"] = window(reqs, gaps, merged, t0 + a, t0 + b)
            out["windows"]["post"] = window(reqs, gaps, merged, t0 + b, t0 + dur)
    return out


def main():
    out_path, specs = sys.argv[1], sys.argv[2:]
    ref = None
    if specs and specs[0].startswith("--ref="):
        ref, specs = specs[0][6:], specs[1:]
    arms = {}
    for s in specs:
        name, dirs = s.split("=", 1)
        arms[name] = [d for d in dirs.split(",") if os.path.exists(os.path.join(d, "requests.jsonl"))]
    offs = None
    if ref and arms.get(ref):
        o = []
        for d in arms[ref]:
            _, t0, _, ev = load(d)
            if "rep_start" in ev and "rep_parked" in ev:
                o.append((ev["rep_start"] - t0, ev["rep_parked"] - t0))
        if o:
            offs = (statistics.mean(x for x, _ in o), statistics.mean(y for _, y in o))
    res = {name: [cycle(d, offs) for d in dirs] for name, dirs in arms.items()}
    res["_baseline_window_offsets"] = offs
    json.dump(res, open(out_path, "w"), indent=1)

    def fmt(v, f="{:.2f}"):
        return "-" if v is None else f.format(v)

    print(f"{'arm':10s} {'cyc':4s} {'repl s':>7s} {'kill->tok':>9s} | replenish window: {'n':>3s} {'TTFT p50/p90':>13s} "
          f"{'ITL p50/p99 ms':>15s} {'stalls':>6s} {'stall s':>7s} {'max':>5s} | post TTFT p50/p90, ITL p50 | err/trunc")
    for name, cs in res.items():
        if name.startswith("_"):
            continue
        for i, c in enumerate(cs, 1):
            w = c["windows"].get("replenish", {}); p = c["windows"].get("post", {})
            print(f"{name:10s} c{i:<3d} {fmt(c.get('replenish_s'), '{:.1f}'):>7s} {fmt(c.get('kill_to_first_token_s'), '{:.1f}'):>9s} | "
                  f"{'':17s} {w.get('n_req', 0):>3d} {fmt(w.get('ttft_p50'))+'/'+fmt(w.get('ttft_p90')):>13s} "
                  f"{fmt(w.get('itl_p50_ms'), '{:.0f}')+'/'+fmt(w.get('itl_p99_ms'), '{:.0f}'):>15s} {w.get('stalls', 0):>6d} "
                  f"{w.get('stall_s', 0):>7.1f} {w.get('max_stall_s', 0):>5.1f} | "
                  f"{fmt(p.get('ttft_p50'))}/{fmt(p.get('ttft_p90'))}, {fmt(p.get('itl_p50_ms'), '{:.0f}')} | {c['errors']}/{c['truncated']}")
            q = c["windows"].get("prestart")
            if q:
                print(f"{'':10s} {'':4s} {fmt(c.get('prestart_s'), '{:.1f}'):>7s} {'(gated spare prestart)':>9s} | {'':17s} "
                      f"{q.get('n_req', 0):>3d} {fmt(q.get('ttft_p50'))+'/'+fmt(q.get('ttft_p90')):>13s} "
                      f"{fmt(q.get('itl_p50_ms'), '{:.0f}')+'/'+fmt(q.get('itl_p99_ms'), '{:.0f}'):>15s} {q.get('stalls', 0):>6d} "
                      f"{q.get('stall_s', 0):>7.1f} {q.get('max_stall_s', 0):>5.1f}")


if __name__ == "__main__":
    main()
