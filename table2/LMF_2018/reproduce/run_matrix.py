import argparse,json,subprocess,sys,os,time
from shared.coe_benchmark import SEEDS,plan_method,save_plan,write_json
from shared.coe_benchmark.artifacts import source_manifest,environment_metadata
from .common import METHOD_DIR,METHOD,source_files
def execute(args):
    plan=plan_method(METHOD,args.run_id,devices=args.devices,phase='formal');root=METHOD_DIR/'exp'/plan['run_id']
    if root.exists():raise FileExistsError('Fresh matrix run-id required; never overwrite')
    save_plan(plan);logs=METHOD_DIR/'log'/plan['run_id'];logs.mkdir(parents=True,exist_ok=True)
    frozen=source_manifest(source_files());env=environment_metadata()
    write_json(root/'frozen_inputs.json',{'sources':frozen,'environment':env})
    state=dict(method=METHOD,run_id=plan['run_id'],status='running',total=len(plan['jobs']),completed=0,replayed=0,failed=[],
               started_at=time.strftime('%Y-%m-%d %H:%M:%S'),policy='author_default_no_hpo',pid=os.getpid(),current_job=None)
    def save():write_json(root/'status.json',state)
    save()
    try:
        for job in plan['jobs']:
            if source_manifest(source_files())!=frozen or environment_metadata()!=env:raise RuntimeError('Frozen source or environment changed')
            d,s=job['dataset'],job['seed'];device=args.devices[0]
            state['current_job']=dict(dataset=d,seed=s,stage='training');save();print(json.dumps({'event':'START',**state['current_job']}),flush=True)
            with (logs/f'{d}_seed{s}_bootstrap.log').open('x') as out:
                subprocess.run([sys.executable,'-u','-m','reproduce.run','--dataset',d,'--seed',str(s),'--device',device,'--phase','formal','--run-id',plan['run_id']],cwd=METHOD_DIR,stdout=out,stderr=subprocess.STDOUT,check=True)
            state['completed']+=1;state['current_job']['stage']='replay';save()
            with (logs/d/f'seed{s}'/'replay.log').open('x') as out:
                subprocess.run([sys.executable,'-u','-m','reproduce.replay','--job-dir',str(root/d/f'seed{s}'),'--device',device],cwd=METHOD_DIR,stdout=out,stderr=subprocess.STDOUT,check=True)
            state['replayed']+=1;save();print(json.dumps({'event':'DONE',**state['current_job'],'completed':state['completed']}),flush=True)
        state.update(status='summarizing',current_job=None);save()
        subprocess.run([sys.executable,'-m','reproduce.summarize','--run-root',str(root)],cwd=METHOD_DIR,check=True)
        state.update(status='complete',finished_at=time.strftime('%Y-%m-%d %H:%M:%S'));save();print(json.dumps(state),flush=True)
    except Exception as exc:
        state['failed'].append({'job':state['current_job'],'error':repr(exc)});state['status']='failed';save();raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--devices',nargs='+',default=['cuda:0']);p.add_argument('--phase',choices=['formal'],default='formal');p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS));a=p.parse_args()
    if tuple(a.seeds)!=SEEDS:raise ValueError('Five registered seeds required')
    execute(a)
