import json, sys, urllib.request
P=["The capital of France is", "Write a Python function that returns the n-th Fibonacci number.\n", "List three prime numbers greater than 100:", "Explain why the sky is blue in one paragraph."]
out=[]
for p in P:
    req=urllib.request.Request("http://localhost:%s/v1/completions" % (sys.argv[2] if len(sys.argv) > 2 else "8000") + "", data=json.dumps({"model":"glm","prompt":p,"max_tokens":48,"temperature":0,"logprobs":1,"seed":0}).encode(), headers={"Content-Type":"application/json"})
    r=json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]
    out.append({"prompt":p,"text":r["text"],"lp":r["logprobs"]["token_logprobs"],"tok":r["logprobs"]["tokens"]})
json.dump(out, open(sys.argv[1],"w")); print([o["text"][:40] for o in out])
