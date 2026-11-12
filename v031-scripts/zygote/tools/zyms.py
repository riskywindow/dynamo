import re,sys
for tag in sys.argv[1:]:
    L=f"/results/zy/{tag}.log"; t0=float(open(L+".launch").read())
    lines=open(L).read().splitlines()
    def first(pat):
        for l in lines:
            if re.search(pat,l):
                try: return round(float(l.split()[0])-t0,1)
                except Exception: pass
        return None
    ms={"api_first": r"\(APIServer pid=\d+\)", "engine_core_first": r"\(EngineCore pid=\d+\)",
        "worker_first": r"\((Worker|VllmWorker)[^)]* pid=\d+\)", "dist_init_rank0": r"world_size=4 rank=0",
        "load_model_start": r"Loading model from scratch", "ready": r"Application startup complete|Starting vLLM API server"}
    print(tag, {k: first(p) for k, p in ms.items()}, flush=True)
