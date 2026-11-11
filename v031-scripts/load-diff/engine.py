"""One vLLM 0.31 engine (offline LLM) for the GMS v0 writer/shadow tests.

--role nogms|writer|reader. Runs the prompt set once ("gen first"), then with
--hold polls /results/ctl/<tag>.cmd for: gen NAME | sleep | wake | exit.
"""
import argparse, json, os, subprocess, sys, time

T0 = time.time()
def log(msg):
    print(f"[T {time.time():.3f} +{time.time()-T0:.1f}s] {msg}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["nogms", "writer", "reader"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--profile", default="glm52-nvfp4-vllm031-gb200-tp4-ep4")
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--no-first-gen", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--extra", default="{}", help="extra LLM kwargs as JSON")
    ap.add_argument("--prompts", default="/results/prompts.json")
    ap.add_argument("--mode", default=None, help="gms_reconstruction_mode, e.g. load_diff")
    ap.add_argument("--model", default="/model-cache/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--no-ep", action="store_true")
    ap.add_argument("--glm-args", type=int, default=1, help="1: GLM Snapshot-arm KV/len args")
    args = ap.parse_args()

    R = "/results"
    MODEL = args.model
    os.makedirs(f"{R}/out", exist_ok=True); os.makedirs(f"{R}/ctl", exist_ok=True)

    from vllm import LLM, SamplingParams

    kw = dict(model=MODEL, tensor_parallel_size=args.tp, enable_expert_parallel=not args.no_ep,
              gpu_memory_utilization=0.45 if args.role != "reader" else 0.3,
              max_num_batched_tokens=8192, max_num_seqs=256,
              enable_prefix_caching=True, enable_chunked_prefill=True, seed=0,
              trust_remote_code=True, enable_flashinfer_autotune=False)
    if args.glm_args:
        kw.update(max_model_len=40960, gpu_memory_utilization=0.8, num_gpu_blocks_override=5641,
                  block_size=64, max_num_seqs=1024)
    else:
        kw.update(max_model_len=8192)
    if args.eager:
        kw["enforce_eager"] = True
    if args.role != "nogms":
        extra = {}
        if args.profile != "none":
            extra["gms_reconstruction_profile"] = args.profile
        if args.mode:
            extra["gms_reconstruction_mode"] = args.mode
        if args.role == "reader":
            extra["gms_read_only"] = True
        kw.update(load_format="gms",
                  worker_cls="gpu_memory_service.integrations.vllm.worker.GMSWorker",
                  model_loader_extra_config=extra)
    kw.update(json.loads(args.extra))
    log(f"LLM kwargs {kw}")
    log(f"env V2={os.environ.get('VLLM_USE_V2_MODEL_RUNNER')} scratch={os.environ.get('DYN_GMS_SCRATCH_KV_ENABLED')}")
    t = time.time()
    llm = LLM(**kw)
    log(f"LLM_READY construct_s={time.time()-t:.1f}")

    prompts = json.load(open(args.prompts))

    def smi(name):
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout
        with open(f"{R}/out/{args.tag}-mem-{name}.txt", "w") as f:
            f.write(f"time {time.time():.3f}\n{out}\n# compute apps\n{procs}")
        log(f"MEM {name}: {out.strip().replace(chr(10), ' | ')}")

    def gen(name, batched=False):
        sps = [SamplingParams(temperature=0.0, max_tokens=p["max_tokens"], logprobs=5, seed=0)
               for p in prompts]
        t = time.time()
        if batched:
            outs = llm.generate([p["prompt"] for p in prompts], sps, use_tqdm=False)
        else:
            # One prompt at a time from an empty prefix cache: batch shapes and
            # CUDA graph sizes are then identical across engines.
            outs = []
            for p, sp in zip(prompts, sps):
                llm.reset_prefix_cache()
                outs += llm.generate([p["prompt"]], [sp], use_tqdm=False)
        dt = time.time() - t
        res = []
        for p, o in zip(prompts, outs):
            c = o.outputs[0]
            lps = []
            for tok, d in zip(c.token_ids, c.logprobs or []):
                lps.append({"tok": tok, "lp": d[tok].logprob if tok in d else None,
                            "top": sorted([[k, v.logprob] for k, v in d.items()], key=lambda x: -x[1])})
            res.append({"name": p["name"], "n_prompt": len(o.prompt_token_ids),
                        "token_ids": list(c.token_ids), "text": c.text, "logprobs": lps})
        json.dump({"tag": args.tag, "gen": name, "batched": batched, "seconds": dt, "results": res},
                  open(f"{R}/out/{args.tag}-{name}.json", "w"))
        log(f"GEN {name} done in {dt:.1f}s: " + " | ".join(f"{r['name']}:{r['text'][:60]!r}" for r in res))

    smi("ready")
    if not args.no_first_gen:
        gen("first")
    if args.hold:
        cmdf, ackf = f"{R}/ctl/{args.tag}.cmd", f"{R}/ctl/{args.tag}.ack"
        log(f"HOLD waiting on {cmdf}")
        while True:
            if not os.path.exists(cmdf):
                time.sleep(0.5); continue
            cmd = open(cmdf).read().strip(); os.remove(cmdf)
            log(f"CMD {cmd}")
            t = time.time()
            try:
                if cmd.startswith("gen "):
                    gen(cmd.split()[1])
                elif cmd.startswith("genb "):
                    gen(cmd.split()[1], batched=True)
                elif cmd == "sleep":
                    llm.sleep(level=1)
                elif cmd == "wake":
                    llm.wake_up()
                elif cmd.startswith("mem "):
                    smi(cmd.split()[1])
                elif cmd == "exit":
                    open(ackf, "a").write(f"exit {time.time():.3f}\n"); break
                status = "ok"
            except Exception as e:
                import traceback; traceback.print_exc(); status = f"error {e!r}"
            dt = time.time() - t
            log(f"CMD_DONE {cmd} {status} {dt:.2f}s")
            open(ackf, "a").write(f"{cmd} {status} {dt:.2f}s {time.time():.3f}\n")
    log("EXIT")
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        log(f"shutdown error {e!r}")
    log("SHUTDOWN_DONE")
    os._exit(0)


if __name__ == "__main__":
    main()
