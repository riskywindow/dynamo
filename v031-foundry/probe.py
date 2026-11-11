#!/usr/bin/env python3
"""Greedy decode probe for the decode-only Foundry port (stdlib only).
Runs the same requests on any arm so native / SAVE / LOAD outputs can be compared token by token:
  solo   : each prompt alone, 32 tokens (batch-1 decode graph)
  pair   : two prompts at once, 32 tokens each (batch-2 decode graph)
  batch8 : eight prompts at once, 24 tokens (larger decode graphs, when captured)
Prompts are natural text (the random-prefix corr prompts are path-sensitive, see the T4 study).
Usage: probe.py --label X --out X.json [--url http://127.0.0.1:8000]
"""
import argparse, json, threading, time, urllib.request

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "Write a haiku about GPUs:",
    "def fibonacci(n):",
    "The three laws of thermodynamics are",
    "List five prime numbers greater than 100:",
    "Translate 'good morning' into Spanish, French and German:",
    "A short story about a lighthouse keeper begins:",
]


def one(url, prompt, mt, out, k):
    body = json.dumps(dict(model="m", prompt=prompt, max_tokens=mt, temperature=0, logprobs=1,
                           ignore_eos=True)).encode()
    t = time.time()
    try:
        req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        lp = r["choices"][0]["logprobs"]
        out[k] = dict(text=r["choices"][0]["text"], tokens=lp["tokens"], lp=lp["token_logprobs"], s=time.time() - t)
    except Exception as e:
        out[k] = dict(error=repr(e)[:300])


def run(url, prompts, mt):
    out = [None] * len(prompts)
    th = [threading.Thread(target=one, args=(url, p, mt, out, i)) for i, p in enumerate(prompts)]
    for t in th: t.start()
    for t in th: t.join()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = dict(label=a.label, first_request_s=None, sets={})
    first = run(a.url, PROMPTS[:1], 8)
    res["first_request_s"] = time.time() - t0
    res["sets"]["solo"] = [run(a.url, [p], 32)[0] for p in PROMPTS[:4]]
    res["sets"]["pair"] = run(a.url, PROMPTS[:2], 32) + run(a.url, PROMPTS[2:4], 32)
    res["sets"]["batch8"] = run(a.url, PROMPTS, 24)
    # ~300-token prompt: a larger breakable prefill graph (PIECEWISE size 512 when captured)
    longp = ("The history of the lighthouse begins with fires on hilltops. " * 30) + "Summarize the passage:"
    res["sets"]["long"] = [run(a.url, [longp], 24)[0], run(a.url, [longp.replace("fires", "beacons")], 24)[0]]
    res["errors"] = sum(1 for v in res["sets"].values() for x in v if "error" in x)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(dict(label=a.label, first_request_s=round(res["first_request_s"], 2), errors=res["errors"],
                          solo0=res["sets"]["solo"][0].get("text", "")[:60])))


if __name__ == "__main__":
    main()
