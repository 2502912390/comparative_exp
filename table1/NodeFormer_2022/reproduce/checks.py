"""Mechanism/gradient checks against official source, no model selection."""
import argparse
import json
import torch
from shared.coe_benchmark import seed_everything, capture_rng, restore_rng
from .common import setup, read_config
from .model import author, build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    setup()
    seed_everything(1024)
    cfg=read_config()
    # Include nonzero relational bias in mechanism checks despite generic rb_order=0.
    # Directed graph with unequal node degrees catches source/target reversal.
    n=17
    edge=torch.stack((torch.cat((torch.arange(n),torch.arange(n),torch.tensor([2,4,9]))),
        torch.cat((torch.arange(n),(torch.arange(n)+3)%n,torch.tensor([1,1,5]))))).to(a.device)
    results=[]
    for training in [False,True]:
        for rb_order in [0,1]:
            tested=dict(cfg,rb_order=rb_order,edge_chunk=7)
            ref=author.NodeFormer(7,32,3,num_heads=4,nb_random_features=30,
                nb_gumbel_sample=10,use_bn=True,use_gumbel=True,use_residual=True,rb_order=rb_order).to(a.device)
            ref.reset_parameters()
            actual=build_model(7,3,tested).to(a.device)
            copied=actual.state_dict()
            copied.update(ref.state_dict())
            actual.load_state_dict(copied)
            x=torch.randn(n,7,device=a.device,requires_grad=True)
            xx=x.detach().clone().requires_grad_(True)
            ref.train(training)
            actual.train(training)
            rng=capture_rng()
            out,links=ref(x,[edge],cfg['tau'])
            target=out.square().mean()-cfg['lamda']*torch.stack(links).mean()
            target.backward()
            restore_rng(rng)
            got,ll=actual(xx,[edge],cfg['tau'])
            loss=got.square().mean()-cfg['lamda']*torch.stack(ll).mean()
            loss.backward()
            torch.testing.assert_close(got,out,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(torch.stack(ll),torch.stack(links),atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(xx.grad,x.grad,atol=4e-6,rtol=1e-3)
            parameters=dict(actual.named_parameters())
            for name,param in ref.named_parameters():
                torch.testing.assert_close(parameters[name].grad,param.grad,atol=4e-6,rtol=1e-3,msg=lambda m:name+': '+m)
            results.append({'training':training,'rb_order':rb_order,'output_max_error':float((got-out).abs().max()),
                'link_max_error':float((torch.stack(ll)-torch.stack(links)).abs().max()),
                'input_grad_max_error':float((xx.grad-x.grad).abs().max())})
    print('WITNESS '+json.dumps({'device':a.device,'gpu':torch.cuda.get_device_name() if 'cuda' in a.device else None,
        'official_output_and_gradients':results,'model':'current_session_model','reviewer':'not_run'}),flush=True)

if __name__=='__main__': main()
