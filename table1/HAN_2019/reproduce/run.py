"""A single HAN job. Development never evaluates or exports test predictions."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from shared.coe_benchmark import (SEEDS, seed_everything, RunPaths, BestCheckpoint,
    load_best, build_manifest, export_result, verify_model_replay, write_json)
from reproduce.common import (METHOD_DIR, DATASETS, SEARCH, registered_config,
    core_files, data_and_graphs, adaptation, reconstruct)
from reproduce.han_model import HAN, official_l2_penalty
from reproduce.tf1_adam import TF1Adam
from shared.coe_benchmark.artifacts import source_manifest
from shared.coe_benchmark.protocol import sha256_file


class Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream
    def write(self, text):
        self.original.write(text); self.stream.write(text); self.stream.flush()
    def flush(self):
        self.original.flush(); self.stream.flush()


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--phase", required=True, choices=("dev","formal","smoke"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-root", default=str(METHOD_DIR / "exp"))
    p.add_argument("--config", type=Path)
    p.add_argument("--candidate", type=int, default=0)
    p.add_argument("--smoke-epochs", type=int)
    p.add_argument("--threads", type=int, default=2)
    return p


def main():
    args = parser().parse_args()
    if Path(args.output_root).resolve() != (METHOD_DIR/"exp").resolve():
        raise ValueError("All outputs must stay in this HAN method exp directory")
    if args.phase == "formal" and (args.seed not in SEEDS or not args.config):
        raise ValueError("Formal seeds and frozen configuration are mandatory")
    if args.phase == "dev" and args.seed != 1024:
        raise ValueError("Development has only registered seed1024")
    if args.smoke_epochs and args.phase != "smoke":
        raise ValueError("Short budgets are allowed only for smoke jobs")
    configs = json.loads(SEARCH.read_text())["candidates"]
    if not 0 <= args.candidate < len(configs):
        raise ValueError("Candidate ID out of registered range")
    config = registered_config(args.config) if args.config else configs[args.candidate]
    if args.config and args.phase == "formal":
        frozen = json.loads(args.config.read_text())
        if (frozen.get("dataset") != args.dataset or frozen.get("selection_basis") != "development_validation_accuracy"
                or frozen.get("development_seed") != 1024):
            raise ValueError("Formal config must be the validation-frozen config for this dataset")
        if frozen.get("source_sha256") != source_manifest(core_files())["sha256"]:
            raise ValueError("Frozen configuration source does not match current method/shared code")
        trials=frozen.get("development_trials",[])
        if len(trials)!=12 or [t["candidate"] for t in trials]!=list(range(12)):
            raise ValueError("Frozen config needs all 12 ordered development candidates")
        for i,trial in enumerate(trials):
            path=Path(trial["job_dir"]).resolve()/"dev_result.json"
            if not path.is_relative_to(METHOD_DIR/"exp") or sha256_file(path)!=trial["dev_result_sha256"]:
                raise ValueError("Development evidence missing, changed, or outside method")
            evidence=json.loads(path.read_text())
            if (evidence["seed"]!=1024 or evidence["dataset"]!=args.dataset or evidence["test_evaluated"]
                    or evidence["config"]!=configs[i] or evidence["best_validation_acc"]!=trial["validation_acc"]):
                raise ValueError("Frozen config evidence not the registered validation-only trial")
        chosen=max(range(12),key=lambda i:trials[i]["validation_acc"])
        if frozen["selected_candidate"]!=chosen or config!=configs[chosen]:
            raise ValueError("Config did not follow validation selection and ordered tie rule")
    config = dict(config)
    if args.smoke_epochs:
        config["epochs"] = args.smoke_epochs
        config["smoke_budget_override"] = True
    job = RunPaths.create(METHOD_DIR,args.run_id,args.dataset,args.seed,args.phase)
    original_stdout,original_stderr=sys.stdout,sys.stderr
    with (job.log/"console.log").open("x",buffering=1) as console:
        sys.stdout, sys.stderr = Tee(sys.stdout,console), Tee(sys.stderr,console)
        try:
            execute(args,config,job)
        except BaseException as error:
            traceback.print_exc()
            write_json(job.exp/"failure.json", {"status":"failed","error":repr(error),"phase":args.phase,
                "dataset":args.dataset,"seed":args.seed,"time":time.time()})
            raise
        finally:
            sys.stdout,sys.stderr=original_stdout,original_stderr


def execute(args,config,job):
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no implicit CPU fallback")
    start = time.perf_counter()
    print(json.dumps({"event":"start","method":"HAN","dataset":args.dataset,"phase":args.phase,
        "seed":args.seed,"device":str(device),"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python":sys.executable,"run_id":job.run_id,"config":config}),flush=True)
    data,features,graphs,supports = data_and_graphs(args.dataset,device)
    print(json.dumps({"event":"data_verified","nodes":data.nodes,"classes":data.classes,
        "views":len(features),"edges":[g.num_edges() for g in graphs],"fingerprints":data.fingerprints}),flush=True)
    model = HAN(data.input_dims,data.classes,config).to(device)
    labels = data.labels.to(device)
    train_mask,val_mask = data.masks["train"].to(device),data.masks["val"].to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    author = json.loads((METHOD_DIR/"sources.json").read_text())
    manifest = build_manifest(data,job,method="HAN",config=config,source_files=core_files(),
        adaptation=adaptation(data,supports),parameter_count=parameter_count,device=device,
        author_commit=author["code_commit"])
    optimizer = TF1Adam(model.parameters(),lr=config["learning_rate"],eps=config["adam_epsilon"])
    selector = BestCheckpoint(job.exp/"best.pt",patience_checks=config["patience_checks"])
    training_start = time.perf_counter()
    print(json.dumps({"event":"training_begin","parameters":parameter_count,"width":256,"heads":8}),flush=True)
    with (job.exp/"training.jsonl").open("x",buffering=1) as trajectory:
        for epoch in range(1,config["epochs"]+1):
            epoch_start = time.perf_counter()
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(features,graphs)
            supervised = F.cross_entropy(logits[train_mask],labels[train_mask])
            penalty = official_l2_penalty(model,config["l2_coefficient"])
            loss = supervised + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            del logits
            model.eval()
            with torch.no_grad():
                val_logits = model(features,graphs)[val_mask]
                validation_acc = (val_logits.argmax(-1)==labels[val_mask]).float().mean().item()
            improved = selector.consider(epoch,validation_acc,{"model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),"config_sha256":manifest["config_sha256"]})
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record = {"epoch":epoch,"loss":loss.item(),"train_ce":supervised.item(),"l2":penalty.item(),
                "validation_acc":validation_acc,"best_validation_acc":selector.best_acc,
                "best_epoch":selector.best_epoch,"improved":improved,
                "seconds":time.perf_counter()-epoch_start,
                "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0}
            trajectory.write(json.dumps(record)+"\n")
            print(json.dumps({"event":"epoch",**record}),flush=True)
            del loss,supervised,penalty,val_logits
            if selector.should_stop:
                print(json.dumps({"event":"early_stop","epoch":epoch}),flush=True); break
    elapsed_training = time.perf_counter()-training_start
    selected = load_best(job.exp/"best.pt")
    model.load_state_dict(selected["payload"]["model_state"],strict=True)
    model.eval()
    with torch.no_grad():
        fresh_val = model(features,graphs)[val_mask]
        restored_acc = (fresh_val.argmax(-1)==labels[val_mask]).float().mean().item()
    if restored_acc != selector.best_acc:
        raise ValueError("Restored best checkpoint validation ACC differs")
    if args.phase == "dev":
        completion = {"status":"complete_development","dataset":args.dataset,"seed":args.seed,
            "candidate":args.candidate,"config":config,"config_sha256":manifest["config_sha256"],
            "best_epoch":selector.best_epoch,"best_validation_acc":selector.best_acc,
            "checkpoint_validation_replayed":True,"test_evaluated":False,
            "training_seconds":elapsed_training,"total_seconds":time.perf_counter()-start}
        write_json(job.exp/"dev_result.json",completion)
        print(json.dumps({"event":"development_complete",**completion}),flush=True)
        return
    with torch.no_grad():
        logits = model(features,graphs).cpu()
    result = export_result(job,data,checkpoint=job.exp/"best.pt",best_epoch=selector.best_epoch,
        best_validation_acc=selector.best_acc,node_ids=data.node_ids,predictions=logits.argmax(-1),
        scores=logits,score_kind="logits",timing={"training_seconds":elapsed_training,"total_seconds":time.perf_counter()-start})
    # Fresh object: never hand saved predictions back to the replay helper.
    replay_model = HAN(data.input_dims,data.classes,config).to(device)
    replay_model.load_state_dict(load_best(job.exp/"best.pt")["payload"]["model_state"],strict=True)
    replay_model.eval()
    with torch.no_grad():
        replay_logits = replay_model(features,graphs).cpu()
    verify_model_replay(job.exp,node_ids=data.node_ids,predictions=replay_logits.argmax(-1),scores=replay_logits)
    write_json(job.exp/"COMPLETED.json", {"status":"complete_single_seed_replay_verified", "run_id":job.run_id})
    print(json.dumps({"event":"complete","test_acc":result["test"]["acc"],"best_epoch":selector.best_epoch,
        "best_validation_acc":selector.best_acc,"model_replay_verified":True}),flush=True)


if __name__ == "__main__":
    main()
