#!/usr/bin/env python3
"""Build <run>/events.json from the collected logs of one fig10-eval Snapshot run. Best effort: missing markers -> null.

Adapted from snapshot-gms-v1-r0.2/scripts/make_events.py (same markers), parameterized by run dir, plus the two
lifecycle spans the Figure 10 rows need:
  phases.container_start = [restore requested (kubectl apply), replacement container started (API startedAt, 1 s res)]
  phases.restore         = [replacement container started, replacement parked on all ranks ('waiting for lock')]
Usage: make_events.py <run-dir>
"""
import datetime as dt, glob, json, re, sys, os

D = sys.argv[1]
L = os.path.join(D, "logs")


def ts(line):  # kubectl --timestamps prefix -> epoch
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z", line)
    if not m:
        return None
    base = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    return base + float(m.group(2) or 0)


def iso(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp() if s else None


def one(pattern):
    g = sorted(glob.glob(os.path.join(L, pattern)))
    return g[0] if g else None


def lines(path):
    try:
        return open(path, errors="replace").read().splitlines()
    except (OSError, TypeError):
        return []


def first(path, pat, after=None):
    for ln in lines(path):
        if re.search(pat, ln):
            t = ts(ln)
            if after is None or (t and t >= after):
                return t
    return None


def all_matches(path, pat):
    return [(ts(ln), ln) for ln in lines(path) if re.search(pat, ln)]


fe = {}
for ln in lines(os.path.join(D, "failover_events.txt")):
    p = ln.split(" ", 2)
    if len(p) >= 2 and re.fullmatch(r"[\d.]+", p[1]):
        fe.setdefault(p[0], (float(p[1]), p[2] if len(p) > 2 else ""))
kill_txt = "\n".join(lines(os.path.join(D, "kill_output.txt")))
m = re.search(r"signal_sent ([\d.]+)", kill_txt)
t_kill = float(m.group(1)) if m else None
m = re.search(r"all_exited_or_zombie ([\d.]+)", kill_txt)
t_procs_dead = float(m.group(1)) if m else None

promoted_log = one("parked-*.log")
serving_log = one("serving-*.log")
rep_log = one("replacement-*.log")
rep_name = os.path.basename(rep_log)[len("replacement-"):-len(".log")] if rep_log else None

t_promoted = first(promoted_log, r"\[Shadow\] Engine awake", after=t_kill) if t_kill else None
# rep_start is taken on the workstation clock right before `kubectl apply` (NTP-synced; see clock_check line:
# recv_minus_node ~0.06 s is the kubectl-exec return latency, so the workstation-node offset is within about that).
t_rep_start = fe["rep_start"][0] if "rep_start" in fe else None
t_rep_ready = first(rep_log, r"waiting for lock") if rep_log else None


def pod_times(role):
    try:
        import yaml
        p = yaml.safe_load(open(os.path.join(L, f"pod-{role}.live.yaml")))
    except Exception:
        return {}
    st = (p.get("status", {}).get("containerStatuses") or [{}])[0].get("state", {})
    started = (st.get("running") or st.get("terminated") or {}).get("startedAt")
    conds = {c["type"]: iso(c.get("lastTransitionTime")) for c in p.get("status", {}).get("conditions", [])}
    return {"created": iso(p["metadata"].get("creationTimestamp")), "container_started": iso(started), "conditions": conds}


def agent_restore(pod, after=None):
    out = {}
    for t, ln in all_matches(os.path.join(L, "agent.log"),
                             re.escape(f"<namespace>/{pod}") + "|" + re.escape(f'"pod": "{pod}"')):
        if after is not None and (t is None or t < after):
            continue
        for key, pat in [("detected", "Restore target detected"), ("start", "=== Starting external restore ==="),
                         ("criu_post_restore", "CRIU post-restore"), ("done", "=== External restore completed ==="),
                         ("summary", "Restore timing summary")]:
            if pat in ln and key not in out:
                out[key] = t
                if key == "summary":
                    i = ln.find('"restore": ')
                    try:
                        out["summary_json"] = json.JSONDecoder().raw_decode(ln[i + len('"restore": '):])[0]
                    except Exception:
                        out["summary_raw"] = ln[i:]
    return out


def dur(s):
    m = re.match(r"([\d.]+)(ms|µs|us|ns|s|m)$", s or "")
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    return v * {"s": 1, "ms": 1e-3, "µs": 1e-6, "us": 1e-6, "ns": 1e-9, "m": 60}[u]


phases = {}
pt = pod_times(rep_name) if rep_name else {}
ag = agent_restore(f"rp-snap-sbs-{rep_name}", after=t_rep_start) if rep_name else {}
cs = pt.get("container_started")
# Figure 10 lifecycle rows (measured): request -> container started -> parked.
if t_rep_start and cs:
    phases["container_start"] = [t_rep_start, cs]
if cs and t_rep_ready:
    phases["restore"] = [cs, t_rep_ready]
# Finer breakdown, as on Nov 7.
if t_rep_start and cs:
    phases["rep.request_to_placeholder_container_started(1s-res)"] = [t_rep_start, cs]
if cs and ag.get("detected"):
    phases["rep.placeholder_standby_to_agent_detect"] = [cs, ag["detected"]]
if ag.get("start") and ag.get("done"):
    phases["rep.agent_external_restore_total"] = [ag["start"], ag["done"]]
    sj = (ag.get("summary_json") or {}).get("phases", {})
    cur = ag["start"]
    for k in ["criu_prepare", "pagebroker_stage", "pagebroker_mount", "overlay_capture", "criu_restore", "gpu_device_map",
              "cuda_restore", "pagebroker_commit"]:
        d = dur(sj.get(k))
        if d is not None:
            phases[f"rep.agent.{k}(duration-anchored)"] = [cur, cur + d]
            cur += d
rs = first(rep_log, r"Restore sentinel detected") if rep_log else None
if ag.get("done") and t_rep_ready:
    phases["rep.post_restore_env_refresh_runtime_to_park"] = [ag["done"], t_rep_ready]
if rs and t_rep_ready:
    phases["rep.engine_sentinel_seen_to_park"] = [rs, t_rep_ready]
if t_rep_ready:
    phases["rep.gms_reattach_and_kv_setup(deferred_to_promotion_upstream)"] = [t_rep_ready, t_rep_ready]


def wake_breakdown(log, after=None):
    la = first(log, r"\[Shadow\] Lock acquired", after=after)
    awake = first(log, r"\[Shadow\] Engine awake", after=after)
    reg = first(log, r"Registering model with endpoint types", after=after)
    wk = [(t, ln) for t, ln in all_matches(log, r"took ([\d.]+) seconds to wake up") if after is None or (t and t >= after)]
    m = re.search(r"took ([\d.]+) seconds to wake up", wk[0][1]) if wk else None
    return {"lock_acquired": la, "engine_awake": awake, "registering_model": reg,
            "vllm_wake_up_s": float(m.group(1)) if m else None}


reqs = []
for x in lines(os.path.join(D, "requests.jsonl")):
    try:
        reqs.append(json.loads(x))
    except ValueError:
        pass
first_tok_after_kill = min([r["t_first"] for r in reqs if r.get("t_first") and not r.get("error")
                            and t_kill and r["t_first"] > t_kill], default=None)
promo = wake_breakdown(promoted_log, after=t_kill) if t_kill else {}
promo["first_successful_token_after_kill"] = first_tok_after_kill
if t_kill and t_procs_dead:
    phases["promo.sigkill_to_all_engine_procs_exited"] = [t_kill, t_procs_dead]
if t_kill and promo.get("lock_acquired"):
    phases["promo.kill_to_lock_acquired"] = [t_kill, promo["lock_acquired"]]
    if promo.get("engine_awake"):
        phases["promo.lock_to_engine_awake"] = [promo["lock_acquired"], promo["engine_awake"]]
if promo.get("engine_awake") and promo.get("registering_model"):
    phases["promo.awake_to_registering"] = [promo["engine_awake"], promo["registering_model"]]
if promo.get("registering_model") and first_tok_after_kill:
    phases["promo.registering_to_first_successful_token"] = [promo["registering_model"], first_tok_after_kill]
phases = {k: v for k, v in phases.items() if v}

ev = {"t_kill": t_kill, "t_promoted": t_promoted, "t_rep_start": t_rep_start, "t_rep_ready": t_rep_ready,
      "phases": phases,
      "t_kill_note": "SIGKILL time from the hostPID killer pod (node clock); t_promoted = '[Shadow] Engine awake'; "
                     "t_rep_ready = replacement '[Shadow] ... waiting for lock' (all ranks restored, engine parked)",
      "t_procs_dead": t_procs_dead, "promotion_wake": promo, "replacement_pod": pt, "replacement_agent": ag,
      "clock_check": fe.get("clock_check", (None, ""))[1]}
json.dump(ev, open(os.path.join(D, "events.json"), "w"), indent=1)
print(json.dumps({k: ev[k] for k in ("t_kill", "t_promoted", "t_rep_start", "t_rep_ready")}, indent=1))
print("replenish_s", (t_rep_ready - t_rep_start) if t_rep_ready and t_rep_start else None)
for k, v in phases.items():
    print(f"  {k}: {v[1]-v[0]:.3f}s")
