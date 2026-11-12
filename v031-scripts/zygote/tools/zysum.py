import re,sys
for pid in sys.argv[2:]:
    txt=open(f"{sys.argv[1]}/{pid}.log").read()
    blocks=txt.split("\n== ")[1:]
    print(f"##### {pid}: {len(blocks)} transitions")
    for b in blocks:
        head=b.splitlines()[0]
        files=[l.strip() for l in b.splitlines() if l.strip().startswith("File") and "sitecustomize" not in l]
        def short(f):
            f=re.sub(r'File "(/usr/local/lib/python3.12/dist-packages/|/results/vllm-site/|/usr/lib/python3.12/)?','',f)
            return f.replace('", line ',':').split(", in ")[0]
        tail=" <- ".join(short(f) for f in files[-5:][::-1])
        m=re.search(r"(\S+ ?\S*): nvidia_fds (\d+)->(\d+) uvm\(cuInit\) (\w+)->(\w+) torch_cuda_init (\w+)->(\w+)",head)
        if m: print(f"  {m.group(1)[:34]:34s} fds {m.group(2):>2}->{m.group(3):<2} uvm {m.group(5):5s} torch {m.group(7):5s} | {tail[:260]}")
