"""Durable, fixed-default 25-job queue with separate-process model replay."""
import argparse,json,subprocess,sys,os,time
from shared.coe_benchmark import plan_method,save_plan
from shared.coe_benchmark.io import write_json
from shared.coe_benchmark.artifacts import source_manifest,environment_metadata
from .common import METHOD_DIR,source_files
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--device',default='cuda:0');o=p.parse_args()
    plan=plan_method('MRiemGNN',o.run_id,devices=[o.device],phase='formal');root=METHOD_DIR/'exp'/plan['run_id']
    if root.exists():raise FileExistsError('Fresh matrix id required')
    save_plan(plan);logs=METHOD_DIR/'log'/plan['run_id'];logs.mkdir(parents=True,exist_ok=True)
    frozen=source_manifest(source_files());env=environment_metadata()
    state={'method':'MRiemGNN','run_id':plan['run_id'],'status':'running','total':len(plan['jobs']),'completed':0,'replayed':0,
           'policy':'author_default_no_hpo','started_at':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':os.getpid(),'current_job':None}
    def save():write_json(root/'status.json',state)
    write_json(root/'frozen_inputs.json',{'sources':frozen,'environment':env});save()
    try:
        for job in plan['jobs']:
            if source_manifest(source_files())!=frozen or environment_metadata()!=env:raise RuntimeError('Frozen core or environment changed')
            dataset,seed=job['dataset'],job['seed'];state['current_job']={'dataset':dataset,'seed':seed,'stage':'training'};save()
            print(json.dumps({'event':'job_start',**state['current_job']}),flush=True)
            with (logs/f'{dataset}_seed{seed}_bootstrap.log').open('w') as out:
                subprocess.check_call([sys.executable,'-u','-m','reproduce.run','--dataset',dataset,'--seed',str(seed),
                    '--device',o.device,'--phase','formal','--run-id',plan['run_id']],cwd=METHOD_DIR,stdout=out,stderr=subprocess.STDOUT)
            state['completed']+=1;state['current_job']['stage']='replay';save()
            log=logs/dataset/f'seed{seed}'/'replay.log'
            with log.open('w') as out:
                subprocess.check_call([sys.executable,'-u','-m','reproduce.replay','--job-dir',str(root/dataset/f'seed{seed}'),
                    '--device',o.device],cwd=METHOD_DIR,stdout=out,stderr=subprocess.STDOUT)
            state['replayed']+=1;save();print(json.dumps({'event':'job_replayed',**state['current_job'],'completed':state['completed']}),flush=True)
        state['status']='summarizing';save()
        subprocess.check_call([sys.executable,'-m','reproduce.summarize','--run-root',str(root)],cwd=METHOD_DIR)
        state.update(status='complete',current_job=None,finished_at=time.strftime('%Y-%m-%d %H:%M:%S'));save();print(json.dumps(state),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));save();raise
if __name__=='__main__':main()
