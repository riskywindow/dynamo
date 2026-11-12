#!/usr/bin/env python3
"""startup_phases.py LOG [LOG...]: phase timeline of one `vllm serve` shadow from its ts.py-prefixed log.

Times are seconds after launch (LOG.launch, written by serve031.sh). The first matching line per marker counts,
except `capture_done` (last graph-capture line). Prints one row per log and a JSON line per log for scripts.
"""
import json, re, sys

MARKERS = [  # (name, regex)
    ("api_up", r"non-default args"),
    ("engine_core", r"Initializing a V1 LLM engine"),
    ("worker_dist", r"world_size=\d+ rank="),
    ("comm_ready", r"all-reduce backends"),
    ("load_start", r"Loading model from scratch"),
    ("attach_done", r"Model loading took"),
    ("gate_hold", r"startup gate: holding"),
    ("gate_release", r"startup gate: released"),
    ("kv_ready", r"Scratch-KV engaged|GPU KV cache size"),
    ("capture_done", r"Graph capturing finished"),
    ("init_done", r"init engine \(profile, create kv cache, warmup model\) took"),
    ("serving", r"Application startup complete"),
]


def phases(path):
    t_launch = float(open(path + ".launch").read().split()[0])
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"(\d+\.\d+) ", line)
        if not m:
            continue
        t = float(m.group(1)) - t_launch
        for name, rx in MARKERS:
            if re.search(rx, line) and (name not in out or name == "capture_done"):
                out[name] = round(t, 1)
    return out


if __name__ == "__main__":
    names = [n for n, _ in MARKERS]
    print("log".ljust(40) + "".join(n[:12].rjust(13) for n in names))
    for p in sys.argv[1:]:
        ph = phases(p)
        print(p[-40:].ljust(40) + "".join(str(ph.get(n, "-")).rjust(13) for n in names))
        print("JSON " + json.dumps({"log": p, **ph}))
