"""Persistent local queue; per-device sequential jobs, automatic replay + summary."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from shared.coe_benchmark import plan_method, save_plan, SEEDS
from shared.coe_benchmark.artifacts import source_manifest
from shared.coe_benchmark.io import write_json
from shared.coe_benchmark.statistics import summarize_run, write_summary
from shared.coe_benchmark.protocol import sha256_file
from .common import METHOD, REPO, read_config, sources


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--devices',nargs='+',default=['cuda:0'])
    p.add_argument('--phase',choices=['formal'],default='formal')
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    a=p.parse_args()
    cfg=read_config()
    plan=plan_method('NodeFormer',a.run_id,devices=a.devices,phase=a.phase,seeds=tuple(a.seeds),python=sys.executable)
    for job in plan['jobs']:
        # Namespaced module invocation avoids all same-name author modules.
        job['argv']=[sys.executable,'-u','-m','table1.NodeFormer_2022.reproduce.run',*job['argv'][2:]]
        job['command']=shlex.join(job['argv'])
    plan['status']='frozen_ready'
    save_plan(plan)
    root=METHOD/'exp'/plan['run_id']
    logs=METHOD/'log'/plan['run_id']
    logs.mkdir(parents=True,exist_ok=True)
    frozen=source_manifest(sources())['files']
    write_json(root/'frozen_config.json',cfg)
    write_json(root/'frozen_sources.json',frozen)
    statuses={}
    lock=threading.Lock()
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
        COE_DATA_ROOT='/home/chenzhichao/Research/CoE2/data/CoE_dataset',
        COE_MULTIVIEW_DATA_ROOT='/home/chenzhichao/Research/CoE2/data/Multi-view datasets')
    def worker(device):
        for job in [j for j in plan['jobs'] if j['device']==device]:
            key=f"{job['dataset']}/seed{job['seed']}"
            try:
                for name,digest in frozen.items():
                    if sha256_file(REPO/name)!=digest:
                        raise RuntimeError(f'Frozen source changed: {name}')
                print(f'START {key} {device}',flush=True)
                with lock:
                    statuses[key]={'status':'running','device':device}
                    write_json(root/'queue_status.json',{'jobs':statuses,'updated_at_unix':time.time()})
                with (logs/f"dispatch_{job['dataset']}_seed{job['seed']}.log").open('w') as stream:
                    rc=subprocess.call(job['argv'],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if rc:
                    raise RuntimeError(f'training exit {rc}; inspect method-local console.log')
                jobdir=root/job['dataset']/f"seed{job['seed']}"
                with (METHOD/'log'/plan['run_id']/job['dataset']/f"seed{job['seed']}"/'replay.log').open('w') as stream:
                    rc=subprocess.call([sys.executable,'-u','-m','table1.NodeFormer_2022.reproduce.replay',
                        '--job-dir',str(jobdir),'--device',device],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if rc:
                    raise RuntimeError(f'independent-process replay exit {rc}')
                status={'status':'complete_replayed'}
                print(f'COMPLETE {key}',flush=True)
            except Exception:
                status={'status':'failed','traceback':traceback.format_exc()}
                print(f'FAILED {key}: {status}',flush=True)
            with lock:
                statuses[key]=status
                write_json(root/'queue_status.json',{'jobs':statuses,'updated_at_unix':time.time()})
                try:
                    write_summary(root,summarize_run(root,expected_datasets=plan['datasets']))
                except Exception:
                    write_json(root/'summary_failure.json',{'traceback':traceback.format_exc()})
    with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
        list(pool.map(worker,a.devices))
    summary=summarize_run(root,expected_datasets=plan['datasets'])
    write_summary(root,summary)
    print('MATRIX_FINISHED '+summary['status'],flush=True)
    if summary['status']!='complete':
        raise SystemExit(1)

if __name__=='__main__': main()
