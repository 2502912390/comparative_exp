"""Durable fixed-default 45-job matrix; each job has its own log and replay."""
import argparse
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from shared.coe_benchmark import SEEDS,plan_method,save_plan,write_json
from .run import METHOD_DIR,DATASETS,core_files
from shared.coe_benchmark.artifacts import source_manifest


def execute(args):
    run_id=args.run_id if args.run_id.startswith('formal_') else 'formal_'+args.run_id
    root=METHOD_DIR/'exp'/run_id
    logroot=METHOD_DIR/'log'/run_id
    if root.exists():raise FileExistsError('Fresh matrix run-id required; never overwrite failures')
    logroot.mkdir(parents=True,exist_ok=True)
    plan=plan_method('InfoMGF',run_id,devices=args.devices,phase='formal')
    save_plan(plan)
    frozen=source_manifest(core_files())
    state={'run_id':run_id,'status':'running','total':45,'completed':0,'replayed':0,'failed':[],
           'started':datetime.now(timezone.utc).isoformat(),'parameter_policy':'author_default_no_hpo',
           'datasets':list(DATASETS),'seeds':list(SEEDS),'sources':frozen}
    def status():write_json(root/'status.json',state)
    status()
    for dataset in DATASETS:
        for seed in SEEDS:
            if source_manifest(core_files())!=frozen:
                state['status']='failed';state['failed'].append({'error':'Core files changed during formal matrix'});status()
                raise RuntimeError('Core files changed during formal matrix')
            device=args.devices[0]
            state['active']={'dataset':dataset,'seed':seed,'stage':'train'};status()
            logs=logroot/dataset/f'seed{seed}'
            argv=[sys.executable,'-u','-m','reproduce.run','--dataset',dataset,'--seed',str(seed),'--device',device,'--phase','formal','--run-id',run_id]
            print('START '+json.dumps(state['active']),flush=True)
            with (logroot/f'{dataset}_seed{seed}_bootstrap.log').open('w') as out:
                code=subprocess.call(argv,cwd=METHOD_DIR,stdout=out,stderr=subprocess.STDOUT)
            if code:
                state['failed'].append({**state['active'],'returncode':code});state['status']='failed';status();return code
            state['completed']+=1;state['active']['stage']='replay';status()
            with (logs/'replay.log').open('w') as out:
                code=subprocess.call([sys.executable,'-u','-m','reproduce.replay','--job-dir',str(root/dataset/f'seed{seed}'),'--device',device],cwd=METHOD_DIR,stdout=out,stderr=subprocess.STDOUT)
            if code:
                state['failed'].append({**state['active'],'returncode':code});state['status']='failed';status();return code
            state['replayed']+=1;status()
            print('DONE '+json.dumps(state['active']),flush=True)
    state['status']='summarizing';state['active']=None;status()
    # Automatically write candidate rows; root evidence record remains parent-owned.
    code=subprocess.call([sys.executable,'-m','reproduce.summarize','--run-root',str(root)],cwd=METHOD_DIR)
    if code:
        state['status']='failed';state['failed'].append({'stage':'summary','returncode':code});status();return code
    state['status']='complete';state['finished']=datetime.now(timezone.utc).isoformat();status()
    print('ALL 45 JOBS AND REPLAYS COMPLETE',flush=True)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--devices',nargs='+',default=['cuda:0']);p.add_argument('--phase',choices=['formal'],default='formal');p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    a=p.parse_args()
    if tuple(a.seeds)!=SEEDS:raise ValueError('Only registered five seeds')
    sys.exit(execute(a))
