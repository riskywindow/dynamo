"""burst.py PORT: large-batch stress on a serving engine (catches post-capture workspace-lock assertions).
Wave 1: 512 concurrent short requests (~200 prompt tokens, 64 out). Wave 2: 48 concurrent ~8K-token prompts (32 out)."""
import json, sys, threading, time, urllib.request
port = sys.argv[1]; url = f"http://127.0.0.1:{port}/v1/completions"
def one(prompt, mt, res, i):
    body = json.dumps(dict(model="glm", prompt=prompt, max_tokens=mt, temperature=0)).encode()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=900).read())
        res[i] = ("ok", r["usage"]["completion_tokens"])
    except Exception as e:
        res[i] = ("err", repr(e)[:200])
def wave(name, n, prompt, mt):
    res = [None] * n; t = time.time()
    th = [threading.Thread(target=one, args=(f"{i} {prompt}", mt, res, i)) for i in range(n)]
    [x.start() for x in th]; [x.join() for x in th]
    errs = [r for r in res if r[0] != "ok"]
    print(f"{name}: {n} requests, {len(errs)} errors, {sum(r[1] for r in res if r[0]=='ok')} tokens, {time.time()-t:.1f} s", errs[:3], flush=True)
short = "Write a short story about a lighthouse keeper who finds a message in a bottle. " * 12
long_ = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 560
wave("short-512", 512, short, 64)
wave("long-48", 48, long_, 32)
