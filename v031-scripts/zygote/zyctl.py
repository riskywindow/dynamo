#!/usr/bin/env python3
"""zyctl.py SOCK LOG [ENV=VAL ...] -- ARGS...: ask the zygote (zygote.py) to fork `vllm ARGS`.

Writes LOG.launch (request time, epoch s) and LOG.pgid (child pid; the child is a session leader), like serve031.sh.
"""
import json
import socket
import sys
import time

sock, logp = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
i = rest.index("--")
env = dict(a.split("=", 1) for a in rest[:i])
args = rest[i + 1:]
with open(logp + ".launch", "w") as f:
    f.write("%.6f\n" % time.time())
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock)
s.sendall((json.dumps({"argv": args, "env": env, "log": logp}) + "\n").encode())
r = json.loads(s.makefile().readline())
with open(logp + ".pgid", "w") as f:
    f.write(str(r["pid"]))
print(json.dumps(r))
