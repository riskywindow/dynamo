#!/usr/bin/env python3
"""Shared open-loop load generator for the replenishment profiling arms.

Stdlib only. Poisson arrivals, 32K-token prompts from the original128 corpus,
1024 forced output tokens, streaming /v1/completions. One JSON line per request.
It never aborts the run on a request error: errors are recorded and it keeps going.

Example:
  python3 loadgen.py --url http://127.0.0.1:8000 --model glm --corpus CORPUS.json \
      --rate 0.35 --duration 720 --out /results/requests.jsonl
Create the --stop-file to stop issuing new requests early (in-flight ones finish).
"""
import argparse, http.client, json, os, random, threading, time, urllib.parse


def build_prompt(corpus, i, vocab_max):
    base = corpus["prompts"][i % len(corpus["prompts"])]
    if i < len(corpus["prompts"]):
        return base
    # Beyond the first 128, make the first 16 tokens unique so no cross-request prefix hits.
    r = random.Random(1_000_003 * i + 7)
    return [r.randrange(1000, vocab_max - 1000) for _ in range(16)] + list(base[16:])


def one_request(url, model, prompt, max_tokens, rec, out_lock, out_f, timeout):
    u = urllib.parse.urlparse(url)
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens, "min_tokens": max_tokens,
        "ignore_eos": True, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    })
    rec["t_send"] = time.time()
    n_chunks = 0
    try:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
        conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        rec["http_status"] = resp.status
        if resp.status != 200:
            rec["error"] = "HTTP %d: %s" % (resp.status, resp.read(500).decode("utf-8", "replace"))
        else:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except Exception:
                    continue
                if "error" in ev:
                    rec["error"] = "SSE error: " + json.dumps(ev["error"])[:500]
                    break
                for ch in ev.get("choices") or []:
                    if ch.get("text"):
                        n_chunks += 1
                        now = time.time()
                        if "t_first" not in rec:
                            rec["t_first"] = now
                            rec["chunk_t"] = []
                        rec["t_last_token"] = now
                        rec["chunk_t"].append(round(now - rec["t_first"], 4))
                if ev.get("usage"):
                    rec["completion_tokens"] = ev["usage"].get("completion_tokens")
                    rec["prompt_tokens"] = ev["usage"].get("prompt_tokens")
        conn.close()
    except Exception as e:  # record and move on
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    rec["t_end"] = time.time()
    rec["n_chunks"] = n_chunks
    with out_lock:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()


def _default_seed():
    """RP_SEED if set; else 20261107 + 101*k when the cwd is a fig10 cycle dir c<k> (distinct Poisson arrivals
    per cycle, so no cycle shares the same arrival gaps); else 20261107."""
    import re
    if os.environ.get("RP_SEED"):
        return int(os.environ["RP_SEED"])
    m = re.fullmatch(r"c(\d+)", os.path.basename(os.getcwd()))
    return 20261107 + 101 * int(m.group(1)) if m else 20261107


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--rate", type=float, default=0.35)
    ap.add_argument("--duration", type=float, default=720.0, help="seconds of arrivals")
    ap.add_argument("--seed", type=int, default=_default_seed())
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=2, help="sequential unmeasured requests before t0")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--stop-file", default="/tmp/loadgen.stop")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    corpus = json.load(open(a.corpus))
    vocab_max = int(corpus.get("vocab_max", 150000))
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    lock = threading.Lock()

    # Warmup: canary prompt, sequential, recorded to a separate file.
    with open(os.path.join(out_dir, "warmup.jsonl"), "a") as wf:
        for k in range(a.warmup):
            rec = {"idx": -1 - k, "kind": "warmup"}
            one_request(a.url, a.model, corpus["canary_prompt"], 64, rec, lock, wf, a.timeout)
            print("warmup", k, "status", rec.get("http_status"), "err", rec.get("error"), flush=True)

    rng = random.Random(a.seed)
    offsets, t = [], 0.0
    while True:
        t += rng.expovariate(a.rate)
        if t > a.duration:
            break
        offsets.append(t)

    t0 = time.time()
    json.dump({"t0": t0, "rate": a.rate, "duration": a.duration, "seed": a.seed,
               "planned_requests": len(offsets), "url": a.url, "model": a.model},
              open(os.path.join(out_dir, "loadgen_start.json"), "w"))
    print("t0", t0, "planned", len(offsets), flush=True)

    threads = []
    with open(a.out, "a") as f:
        for i, off in enumerate(offsets):
            delay = t0 + off - time.time()
            if delay > 0:
                time.sleep(delay)
            if os.path.exists(a.stop_file):
                print("stop file seen; no new requests after idx", i, flush=True)
                break
            rec = {"idx": i, "kind": "measured", "sched_offset": off, "input_tokens": 32768}
            th = threading.Thread(target=one_request, daemon=True,
                                  args=(a.url, a.model, build_prompt(corpus, i, vocab_max), a.max_tokens,
                                        rec, lock, f, a.timeout))
            th.start()
            threads.append(th)
        deadline = time.time() + a.timeout
        for th in threads:
            th.join(max(0.0, deadline - time.time()))
    print("done; issued", len(threads), flush=True)


if __name__ == "__main__":
    main()
