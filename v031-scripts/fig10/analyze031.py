#!/usr/bin/env python3
"""analyze031.py ARMS_DIR [--ref ARM]: serving latency per arm in the replenish window vs before it.

Per arm (ARMS_DIR/<arm>/): requests.jsonl (loadgen_perchunk), loadgen_start.json, events.txt, shadow.log
(epoch-ms prefixed by ts.py). Window = [shadow launch, shadow parked]. Arms without a shadow (baseline) use the
window offsets (relative to t0) of --ref ARM. "pre" = [t0 + 30 s, window start).
Stalls: inter-chunk gaps > 0.5 s of streams in decode, merged across streams; each is attributed to the shadow
phase (last phase marker in shadow.log at or before the stall start).
"""
import argparse, json, os, statistics

STALL = 0.5
MARKERS = [  # (phase name, substring of the first log line that starts it)
    ("dist_init", "world_size="),
    ("weight_attach", "Loading model from scratch"),
    ("profile_run", "Model loading took"),
    ("kv_init", "[GMS] projected available memory"),
    ("kernel_warmup", "JIT kernel warmup starting"),
    ("warmup_kernels+graphs", "Skipping FlashInfer autotune"),
    ("post_graphs", "Graph capturing finished"),
    ("api_start", "init engine"),
    ("api_ready", "Application startup complete"),
]


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def load_arm(d):
    reqs = [json.loads(l) for l in open(os.path.join(d, "requests.jsonl")) if l.strip()]
    t0 = json.load(open(os.path.join(d, "loadgen_start.json")))["t0"]
    ev = {}
    if os.path.exists(os.path.join(d, "events.txt")):
        for l in open(os.path.join(d, "events.txt")):
            k, t = l.split()
            ev.setdefault(k, float(t))
    phases = []
    log = os.path.join(d, "shadow.log")
    if os.path.exists(log):
        seen = set()
        for l in open(log, errors="replace"):
            ts, _, rest = l.partition(" ")
            try:
                ts = float(ts)
            except ValueError:
                continue
            for name, sub in MARKERS:
                if name not in seen and sub in rest:
                    if name == "post_graphs":  # last capture line, not first
                        continue
                    seen.add(name)
                    phases.append((ts, name))
            if "Graph capturing finished" in rest:
                phases = [p for p in phases if p[1] != "post_graphs"] + [(ts, "post_graphs")]
        if "launch" in ev:
            phases.append((ev["launch"], "spawn+imports"))
        if "parked" in ev:
            phases.append((ev["parked"], "parked"))
        phases.sort()
    return reqs, t0, ev, phases


def stalls(reqs):
    gaps = []
    for r in reqs:
        if r.get("kind") != "measured" or "t_first" not in r:
            continue
        times = [r["t_first"] + c for c in r.get("chunk_t", [])]
        for a, b in zip(times, times[1:]):
            gaps.append((a, b))
    big = sorted((a, b) for a, b in gaps if b - a > STALL)
    merged = []
    for a, b in big:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return gaps, merged


def window_stats(reqs, gaps, merged, a, b):
    ttft = [r["t_first"] - r["t_send"] for r in reqs
            if r.get("kind") == "measured" and "t_first" in r and a <= r["t_send"] < b]
    itl = [y - x for x, y in gaps if a <= y < b]
    st = [(s, e) for s, e in merged if a <= s < b]
    return {"n_req": len(ttft), "ttft_p50": pct(ttft, 50), "ttft_p90": pct(ttft, 90),
            "itl_p50": pct(itl, 50), "itl_p90": pct(itl, 90), "itl_p99": pct(itl, 99),
            "n_stalls": len(st), "stall_s": round(sum(e - s for s, e in st), 2),
            "max_stall": round(max((e - s for s, e in st), default=0), 2)}


def phase_at(phases, t):
    name = "before_launch"
    for ts, n in phases:
        if ts <= t:
            name = n
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms_dir")
    ap.add_argument("--ref", default=None, help="arm whose window offsets the no-shadow arms use")
    a = ap.parse_args()
    arms = sorted(x for x in os.listdir(a.arms_dir) if os.path.exists(os.path.join(a.arms_dir, x, "requests.jsonl")))
    data = {x: load_arm(os.path.join(a.arms_dir, x)) for x in arms}
    ref = a.ref or next((x for x in arms if "launch" in data[x][2]), None)
    rw = None
    if ref:
        _, t0r, evr, _ = data[ref]
        rw = (evr["launch"] - t0r, evr.get("parked", evr.get("ready", evr["launch"] + 90)) - t0r)
    out = {}
    print(f"{'arm':14s} {'window':>13s} {'n':>3s} {'TTFT p50/p90 s':>16s} {'ITL p50/p90/p99 ms':>20s} {'stalls':>6s} {'stall s':>7s} {'max':>5s}")
    for x in arms:
        reqs, t0, ev, phases = data[x]
        gaps, merged = stalls(reqs)
        if "launch" in ev:
            w = (ev["launch"], ev.get("parked", ev.get("ready", ev["launch"] + 90)))
        else:
            w = (t0 + rw[0], t0 + rw[1]) if rw else (t0 + 120, t0 + 210)
        res = {"pre": window_stats(reqs, gaps, merged, t0 + 30, w[0]),
               "window": window_stats(reqs, gaps, merged, w[0], w[1]),
               "window_s": round(w[1] - w[0], 1),
               "errors": sum(1 for r in reqs if r.get("error")),
               "events": {k: round(v - t0, 2) for k, v in ev.items()},
               "phases": [(round(ts - w[0], 2), n) for ts, n in phases],
               "stalls_in_window": [(round(s - w[0], 2), round(e - s, 2), phase_at(phases, s))
                                    for s, e in merged if w[0] <= s < w[1]]}
        out[x] = res
        for k in ("pre", "window"):
            s = res[k]
            f = lambda v, m=1: "-" if v is None else f"{v * m:.2f}" if m == 1 else f"{v * m:.0f}"
            print(f"{x:14s} {k + (' %.0fs' % res['window_s'] if k == 'window' else ''):>13s} {s['n_req']:3d} "
                  f"{f(s['ttft_p50']) + '/' + f(s['ttft_p90']):>16s} "
                  f"{f(s['itl_p50'], 1000) + '/' + f(s['itl_p90'], 1000) + '/' + f(s['itl_p99'], 1000):>20s} "
                  f"{s['n_stalls']:6d} {s['stall_s']:7.2f} {s['max_stall']:5.2f}")
    for x in arms:
        r = out[x]
        if r["phases"]:
            print(f"\n{x}: phases (s from launch): " + ", ".join(f"{n}@{t}" for t, n in r["phases"]))
        if r["stalls_in_window"]:
            print(f"{x}: stalls (start s from launch, duration s, shadow phase): {r['stalls_in_window']}")
    json.dump(out, open(os.path.join(a.arms_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
