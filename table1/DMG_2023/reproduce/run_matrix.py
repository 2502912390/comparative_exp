"""Serial durable fixed-default 25-run matrix; stops visibly on first failure."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "reproduce"

import argparse,json,os,subprocess,sys,time
from pathlib import Path
from shared.coe_benchmark import plan_method,save_plan
from shared.coe_benchmark.io import write_json
from .common import METHOD_DIR,DATASETS,source_files,resolve_config
from shared.coe_benchmark.artifacts import source_manifest,environment_metadata
from shared.coe_benchmark.protocol import json_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--device',default='cuda:0');o=p.parse_args()
    plan=plan_method('DMG',o.run_id,devices=[o.device],phase='formal');save_plan(plan)
    run_id=plan['run_id'];root=METHOD_DIR/'exp'/run_id;root.mkdir(parents=True,exist_ok=True)
    frozen={'sources':source_manifest(source_files()),'configs':{d:resolve_config(d) for d in DATASETS},'environment':environment_metadata()}
    write_json(root/'frozen_inputs.json',frozen)
    def check_frozen():
        if source_manifest(source_files())!=frozen['sources']:raise RuntimeError('Core sources changed after matrix freeze')
        if {d:resolve_config(d) for d in DATASETS}!=frozen['configs']:raise RuntimeError('Fixed configuration changed')
        if environment_metadata()!=frozen['environment']:raise RuntimeError('Environment changed after matrix freeze')
    state={'method':'DMG','run_id':run_id,'status':'running','total':25,'completed':0,'replayed':0,'policy':'author_default_no_hpo','pid':os.getpid(),'started_at':time.strftime('%Y-%m-%d %H:%M:%S'),'current_job':None}
    def save():write_json(root/'status.json',state)
    save()
    for dataset in DATASETS:
        for seed in range(1024,1029):
            try:check_frozen()
            except Exception as exc:
                state.update(status='failed_frozen_inputs',error=str(exc));save();raise
            state['current_job']={'dataset':dataset,'seed':seed,'stage':'training'};save()
            print(json.dumps({'event':'job_start',**state['current_job']}),flush=True)
            argv=[sys.executable,'-u','-m','reproduce.run','--dataset',dataset,'--seed',str(seed),'--device',o.device,'--phase','formal','--run-id',run_id]
            returncode=subprocess.call(argv,cwd=METHOD_DIR)
            if returncode:state.update(status='failed',failed_argv=argv,returncode=returncode);save();raise SystemExit(returncode)
            state['completed']+=1;state['current_job']['stage']='replay';save()
            job=root/dataset/f'seed{seed}';log=METHOD_DIR/'log'/run_id/dataset/f'seed{seed}'/'replay.log'
            with log.open('w') as stream:
                argv=[sys.executable,'-u','-m','reproduce.replay','--job-dir',str(job),'--device',o.device]
                returncode=subprocess.call(argv,cwd=METHOD_DIR,stdout=stream,stderr=subprocess.STDOUT)
            if returncode:state.update(status='failed_replay',failed_argv=argv,returncode=returncode);save();raise SystemExit(returncode)
            state['replayed']+=1;save();print(json.dumps({'event':'job_replayed','dataset':dataset,'seed':seed,'completed':state['completed']}),flush=True)
    subprocess.check_call([sys.executable,'-m','reproduce.summarize','--run-root',str(root)],cwd=METHOD_DIR)
    state.update(status='complete',current_job=None,finished_at=time.strftime('%Y-%m-%d %H:%M:%S'));save();print(json.dumps(state),flush=True)
if __name__=='__main__':main()
