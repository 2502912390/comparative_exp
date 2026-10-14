"""Validation-only 12-candidate development, freeze, 25 formal jobs and replay.

No summary or table row is automatically generated: user triggers it later.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.coe_benchmark import SEEDS, write_json, recount_result
from shared.coe_benchmark.artifacts import source_manifest
from shared.coe_benchmark.protocol import json_digest, sha256_file
from reproduce.common import METHOD_DIR, DATASETS, SEARCH, core_files


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True)
    p.add_argument("--devices",nargs="+",default=["cuda:0"])
    p.add_argument("--phase",default="formal",choices=["formal"])
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    if args.seeds != list(SEEDS) or args.devices != ["cuda:0"]:
        raise ValueError("Matrix uses exact five seeds and one allocated logical GPU cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("This HAN batch must bind physical GPU1 with CUDA_VISIBLE_DEVICES=1")
    batch=args.run_id
    if not batch.replace("_","").replace("-","").isalnum():
        raise ValueError("Unsafe batch ID")
    formal_id=batch if batch.startswith("formal_") else "formal_"+batch
    dev_base="dev_"+batch.removeprefix("formal_")
    run_root=METHOD_DIR/"exp"/formal_id
    control_log=METHOD_DIR/"log"/formal_id
    if run_root.exists():
        raise FileExistsError("Use a fresh batch ID; existing failed/completed batches are preserved")
    run_root.mkdir(parents=True);control_log.mkdir(parents=True,exist_ok=True)
    task_logs=control_log/"tasks";task_logs.mkdir()
    configs=json.loads(SEARCH.read_text())["candidates"]
    if len(configs)!=12 or len({json_digest(c) for c in configs})!=12:
        raise ValueError("Exactly 12 distinct pre-registered development candidates required")
    sources=source_manifest(core_files())
    plan={"method":"HAN","protocol_id":"coe1005_fixedsplit_v1.1","batch_id":batch,
        "run_id":formal_id,"phase":"formal","datasets":list(DATASETS),"expected_datasets":list(DATASETS),
        "seeds":list(SEEDS),"development_seed":1024,"candidates":configs,
        "formal_job_count":25,"development_job_count":60,"automatic_summary":False,
        "allocated_physical_gpu":1,"devices":args.devices,"source_lock":sources,
        "jobs":[{"dataset":dataset,"seed":seed,"phase":"formal","run_id":formal_id}
                for dataset in DATASETS for seed in SEEDS]}
    write_json(run_root/"planned_jobs.json",plan)
    write_json(run_root/"source_lock.json",sources)
    frozen_root=run_root/"frozen_configs";frozen_root.mkdir()
    state={"status":"running","started_unix":time.time(),"formal_id":formal_id,
        "completed_development":0,"completed_formal":0,"current":None}
    write_status(run_root,state)
    try:
        for dataset in DATASETS:
            trials=[]
            for candidate in range(12):
                ensure_source_lock(sources)
                dev_id=f"{dev_base}_c{candidate:02d}"
                state["current"]={"phase":"dev","dataset":dataset,"candidate":candidate,"seed":1024}
                write_status(run_root,state)
                argv=[sys.executable,"-u","-m","reproduce.run","--dataset",dataset,"--seed","1024",
                    "--device","cuda:0","--phase","dev","--run-id",dev_id,
                    "--candidate",str(candidate),"--threads",str(args.threads)]
                execute(argv,task_logs/f"dev_{dataset}_c{candidate:02d}.log")
                job=METHOD_DIR/"exp"/dev_id/dataset/"seed1024"
                result=json.loads((job/"dev_result.json").read_text())
                if (result["status"]!="complete_development" or result["test_evaluated"]
                        or result["config"]!=configs[candidate]):
                    raise ValueError("Development evidence invalid")
                trials.append({"candidate":candidate,"validation_acc":result["best_validation_acc"],
                    "best_epoch":result["best_epoch"],"job_dir":str(job),
                    "dev_result_sha256":sha256_file(job/"dev_result.json")})
                state["completed_development"]+=1;write_status(run_root,state)
                print(json.dumps({"event":"development_trial_complete","dataset":dataset,**trials[-1]}),flush=True)
            best=max(range(12),key=lambda i:trials[i]["validation_acc"])
            frozen={"dataset":dataset,"config":configs[best],"selected_candidate":best,
                "selection_basis":"development_validation_accuracy","development_seed":1024,
                "tie_break":"first registered candidate","development_trials":trials,
                "source_sha256":sources["sha256"],"protocol_id":"coe1005_fixedsplit_v1.1"}
            frozen_path=frozen_root/f"{dataset}.json";write_json(frozen_path,frozen)
            print(json.dumps({"event":"configuration_frozen","dataset":dataset,"selected_candidate":best,
                "validation_acc":trials[best]["validation_acc"],"path":str(frozen_path)}),flush=True)
            for seed in SEEDS:
                ensure_source_lock(sources)
                state["current"]={"phase":"formal","dataset":dataset,"seed":seed}
                write_status(run_root,state)
                argv=[sys.executable,"-u","-m","reproduce.run","--dataset",dataset,"--seed",str(seed),
                    "--device","cuda:0","--phase","formal","--run-id",formal_id,
                    "--config",str(frozen_path),"--threads",str(args.threads)]
                execute(argv,task_logs/f"formal_{dataset}_seed{seed}.log")
                job=run_root/dataset/f"seed{seed}"
                replay_argv=[sys.executable,"-u","-m","reproduce.replay","--job-dir",str(job),
                    "--device","cuda:0","--threads",str(args.threads)]
                execute(replay_argv,task_logs/f"replay_{dataset}_seed{seed}.log")
                checked=recount_result(job)
                if not checked["result"]["model_replay_verified"]:
                    raise ValueError("Formal job lacks independent-process checkpoint replay")
                state["completed_formal"]+=1;write_status(run_root,state)
                print(json.dumps({"event":"formal_seed_complete","dataset":dataset,"seed":seed,
                    "completed_formal":state["completed_formal"]}),flush=True)
        state.update(status="completed_awaiting_user_summary",current=None,finished_unix=time.time())
        write_status(run_root,state);write_json(run_root/"COMPLETED.json",state)
        print(json.dumps({"event":"MATRIX_COMPLETED","formal_jobs":25,"development_jobs":60,
            "summary":"not_run_user_will_trigger","run_root":str(run_root)}),flush=True)
    except BaseException as error:
        state.update(status="failed",error=repr(error),finished_unix=time.time())
        write_status(run_root,state);write_json(run_root/"FAILED.json",state)
        print(json.dumps({"event":"MATRIX_FAILED",**state}),flush=True)
        raise


def write_status(run_root,state):
    write_json(run_root/"progress.json",state)
    write_json(run_root/"status.json",state)


def ensure_source_lock(expected):
    if source_manifest(core_files()) != expected:
        raise ValueError("Core/shared source changed after batch registration; refusing mixed-code seeds")


def execute(argv,path):
    print(json.dumps({"event":"launch","argv":argv,"log":str(path)}),flush=True)
    with path.open("x",buffering=1) as stream:
        completed=subprocess.run(argv,cwd=METHOD_DIR,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Task exited {completed.returncode}; retained log: {path}")


if __name__=="__main__":main()
