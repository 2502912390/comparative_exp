"""Fixed author-default HAN job; full graph, no hyperparameter search."""
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
from reproduce.common import (METHOD_DIR, DATASETS, DEFAULT, registered_config,
    core_files, data_and_graphs, adaptation)
from reproduce.han_model import HAN, official_l2_penalty
from reproduce.tf1_adam import TF1Adam
from reproduce.inference import StableInference
from reproduce.stopping import OfficialEarlyStopping


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
    p.add_argument("--phase", required=True, choices=("formal","smoke"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-root", default=str(METHOD_DIR / "exp"))
    p.add_argument("--config", type=Path,default=DEFAULT)
    p.add_argument("--smoke-epochs", type=int)
    p.add_argument("--threads", type=int, default=2)
    return p


def main():
    args = parser().parse_args()
    if Path(args.output_root).resolve() != (METHOD_DIR/"exp").resolve():
        raise ValueError("All outputs must stay in this HAN method exp directory")
    if args.phase == "formal" and args.seed not in SEEDS:
        raise ValueError("Formal seeds are the fixed five")
    if args.smoke_epochs and args.phase != "smoke":
        raise ValueError("Short budgets are allowed only for smoke jobs")
    config = dict(registered_config(args.config))
    if args.smoke_epochs:
        if args.smoke_epochs<1:raise ValueError("Smoke budget must be positive")
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
    torch.set_num_threads(args.threads);seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no implicit CPU fallback")
    start = time.perf_counter()
    print(json.dumps({"event":"start","method":"HAN","dataset":args.dataset,"phase":args.phase,
        "seed":args.seed,"device":str(device),"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python":sys.executable,"run_id":job.run_id,"config":config,"hyperparameter_search":False}),flush=True)
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
    stopping=OfficialEarlyStopping(config["patience_checks"])
    # Independent weights/features copy: never cast canonical tensors or training model in place.
    stable=StableInference(model,features,graphs)
    validation_seconds=0.0;training_start = time.perf_counter()
    print(json.dumps({"event":"training_begin","parameters":parameter_count,"width":64,"heads":8,
        "training_dtype":"float32","validation_export_replay_dtype":"float64","inference_device":str(device)}),flush=True)
    with (job.exp/"training.jsonl").open("x",buffering=1) as trajectory:
        for epoch in range(1,config["epochs"]+1):
            epoch_start = time.perf_counter()
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(features,graphs)
            supervised = F.cross_entropy(logits[train_mask],labels[train_mask])
            penalty = official_l2_penalty(model,config["l2_coefficient"])
            loss = supervised + penalty
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
            loss.backward();optimizer.step();del logits
            eval_start=time.perf_counter();val_logits=stable(model)[val_mask]
            validation_acc = (val_logits.argmax(-1)==labels[val_mask]).float().mean().item()
            validation_loss=F.cross_entropy(val_logits,labels[val_mask]).item()
            should_stop=stopping.consider(validation_acc,validation_loss)
            if device.type=="cuda":torch.cuda.synchronize(device)
            eval_seconds=time.perf_counter()-eval_start;validation_seconds+=eval_seconds
            improved = selector.consider(epoch,validation_acc,{"model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),"config_sha256":manifest["config_sha256"]})
            record = {"epoch":epoch,"loss":loss.item(),"train_ce":supervised.item(),"l2":penalty.item(),
                "validation_acc":validation_acc,"best_validation_acc":selector.best_acc,
                "best_epoch":selector.best_epoch,"improved":improved,
                "validation_ce":validation_loss,"native_curr_step":stopping.curr_step,
                "seconds":time.perf_counter()-epoch_start,"validation_seconds":eval_seconds,
                "validation_dtype":"float64",
                "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0}
            trajectory.write(json.dumps(record)+"\n");print(json.dumps({"event":"epoch",**record}),flush=True)
            del loss,supervised,penalty,val_logits
            if should_stop:
                print(json.dumps({"event":"early_stop","epoch":epoch}),flush=True);break
    elapsed_training = time.perf_counter()-training_start
    selected = load_best(job.exp/"best.pt")
    model.load_state_dict(selected["payload"]["model_state"],strict=True)
    restored_acc=(stable(model)[val_mask].argmax(-1)==labels[val_mask]).float().mean().item()
    if restored_acc != selector.best_acc:raise ValueError("Restored best validation ACC differs")
    inference_start=time.perf_counter();logits=stable(model).cpu()
    inference_seconds=time.perf_counter()-inference_start
    result = export_result(job,data,checkpoint=job.exp/"best.pt",best_epoch=selector.best_epoch,
        best_validation_acc=selector.best_acc,node_ids=data.node_ids,predictions=logits.argmax(-1),
        scores=logits,score_kind="logits",timing={"training_including_validation_seconds":elapsed_training,
            "validation_seconds":validation_seconds,"final_inference_seconds":inference_seconds,
            "total_seconds":time.perf_counter()-start,"training_dtype":"float32",
            "validation_export_replay_dtype":"float64","inference_device":str(device)})
    # Fresh object and fresh inference copy, no saved prediction array as replay input.
    replay_model = HAN(data.input_dims,data.classes,config).to(device)
    replay_model.load_state_dict(load_best(job.exp/"best.pt")["payload"]["model_state"],strict=True)
    replay_logits=StableInference(replay_model,features,graphs)(replay_model).cpu()
    verify_model_replay(job.exp,node_ids=data.node_ids,predictions=replay_logits.argmax(-1),scores=replay_logits)
    write_json(job.exp/"COMPLETED.json", {"status":"complete_single_seed_replay_verified", "run_id":job.run_id})
    print(json.dumps({"event":"complete","test_acc":result["test"]["acc"],"best_epoch":selector.best_epoch,
        "best_validation_acc":selector.best_acc,"model_replay_verified":True,"score_dtype":"float64"}),flush=True)


if __name__ == "__main__":main()
