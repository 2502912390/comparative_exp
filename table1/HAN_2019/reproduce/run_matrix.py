"""Fixed author-default25 formal jobs and independent-process checkpoint replay.

No development search, summaries, or paper edits. First complete ACM task is
regression gate: its training and replay must finish before later jobs.
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
from shared.coe_benchmark import SEEDS, PROTOCOL_ID, write_json, recount_result
from shared.coe_benchmark.artifacts import source_manifest
from shared.coe_benchmark.protocol import sha256_file
from reproduce.common import METHOD_DIR, DATASETS, DEFAULT, core_files,registered_config


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True)
    p.add_argument("--devices",nargs="+",default=["cuda:0"])
    p.add_argument("--phase",default="formal",choices=["formal"])
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    if args.seeds != list(SEEDS) or args.devices != ["cuda:0"]:
        raise ValueError("Matrix uses exact five seeds and allocated logicalGPU cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("HAN must bind physicalGPU1 with CUDA_VISIBLE_DEVICES=1")
    batch=args.run_id
    if not batch.replace("_","").replace("-","").isalnum():raise ValueError("Unsafe batchID")
    formal_id=batch if batch.startswith("formal_") else "formal_"+batch
    run_root=METHOD_DIR/"exp"/formal_id;control_log=METHOD_DIR/"log"/formal_id
    if run_root.exists():raise FileExistsError("Use freshbatchID; old failed evidence is preserved")
    config=registered_config(DEFAULT);sources=source_manifest(core_files())
    run_root.mkdir(parents=True);control_log.mkdir(parents=True,exist_ok=True)
    task_logs=control_log/"tasks";task_logs.mkdir()
    plan={"method":"HAN","protocol_id":PROTOCOL_ID,"batch_id":batch,"run_id":formal_id,
        "phase":"formal","datasets":list(DATASETS),"expected_datasets":list(DATASETS),
        "seeds":list(SEEDS),"configuration":config,"config_source":str(DEFAULT),
        "config_source_sha256":sha256_file(DEFAULT),"comparison_policy":"author_defaults_no_hpo_v1",
        "formal_job_count":25,"development_job_count":0,"hyperparameter_search":False,
        "automatic_summary":False,"allocated_physical_gpu":1,"devices":args.devices,"source_lock":sources,
        "jobs":[{"dataset":dataset,"seed":seed,"phase":"formal","run_id":formal_id}
                for dataset in DATASETS for seed in SEEDS]}
    write_json(run_root/"planned_jobs.json",plan);write_json(run_root/"source_lock.json",sources)
    state={"status":"running","started_unix":time.time(),"formal_id":formal_id,
        "comparison_policy":"author_defaults_no_hpo_v1","completed_development":0,
        "completed_formal":0,"current":None,"first_full_acm_replay_gate":"pending"}
    write_status(run_root,state)
    print(json.dumps({"event":"FIXED_DEFAULT_MATRIX_STARTED","formal_jobs":25,"development_jobs":0,
        "width":64,"hyperparameter_search":False}),flush=True)
    try:
        for dataset in DATASETS:
            for seed in SEEDS:
                if source_manifest(core_files())!=sources:
                    raise ValueError("Method/shared core changed; refusing mixed-source seeds")
                state["current"]={"phase":"formal","dataset":dataset,"seed":seed};write_status(run_root,state)
                argv=[sys.executable,"-u","-m","reproduce.run","--dataset",dataset,"--seed",str(seed),
                    "--device","cuda:0","--phase","formal","--run-id",formal_id,
                    "--config",str(DEFAULT),"--threads",str(args.threads)]
                execute(argv,task_logs/f"formal_{dataset}_seed{seed}.log")
                job=run_root/dataset/f"seed{seed}"
                execute([sys.executable,"-u","-m","reproduce.replay","--job-dir",str(job),
                    "--device","cuda:0","--threads",str(args.threads)],task_logs/f"replay_{dataset}_seed{seed}.log")
                checked=recount_result(job)
                if not checked["result"]["model_replay_verified"]:
                    raise ValueError("Formal job lacks fresh process checkpoint replay")
                if dataset=="acm" and seed==1024:
                    state["first_full_acm_replay_gate"]="passed"
                    write_json(run_root/"ACM_REPLAY_GATE_PASSED.json",{
                        "dataset":dataset,"seed":seed,"best_epoch":checked["result"]["best_epoch"],
                        "status":"verified_complete_full_formal_training",
                        "replay":"fresh_object_and_independent_process_float64_scores"})
                state["completed_formal"]+=1;write_status(run_root,state)
                print(json.dumps({"event":"formal_seed_complete","dataset":dataset,"seed":seed,
                    "completed_formal":state["completed_formal"]}),flush=True)
        state.update(status="completed_awaiting_user_summary",current=None,finished_unix=time.time())
        write_status(run_root,state);write_json(run_root/"COMPLETED.json",state)
        print(json.dumps({"event":"MATRIX_COMPLETED","formal_jobs":25,"development_jobs":0,
            "summary":"not_run_user_will_trigger","run_root":str(run_root)}),flush=True)
    except BaseException as error:
        state.update(status="failed",error=repr(error),finished_unix=time.time())
        write_status(run_root,state);write_json(run_root/"FAILED.json",state)
        print(json.dumps({"event":"MATRIX_FAILED",**state}),flush=True);raise


def write_status(root,state):
    write_json(root/"progress.json",state);write_json(root/"status.json",state)


def execute(argv,path):
    print(json.dumps({"event":"launch","argv":argv,"log":str(path)}),flush=True)
    with path.open("x",buffering=1) as stream:
        completed=subprocess.run(argv,cwd=METHOD_DIR,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError(f"Task exited{completed.returncode}; retainedlog:{path}")


if __name__=="__main__":main()
