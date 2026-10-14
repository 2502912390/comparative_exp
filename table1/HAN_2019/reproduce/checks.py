"""Native64 HAN FP32/FP64 sparse-vs-dense outputs and gradients on directed relations."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from shared.coe_benchmark import seed_everything,write_json
from reproduce.common import METHOD_DIR,DEFAULT,registered_config
from reproduce.han_model import HAN,graph_from_adjacency,official_l2_penalty
from reproduce.tf1_adam import TF1Adam
from reproduce.inference import StableInference
from reproduce.stopping import OfficialEarlyStopping


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");args=p.parse_args()
    torch.set_num_threads(2);seed_everything(1024);config=registered_config(DEFAULT)
    pairs=[[(0,1),(1,2),(2,0),(3,1)],[(0,2),(2,3),(3,0),(1,3)]]
    graphs=[];dense=[]
    for pair in pairs:
        index=torch.tensor(pair).T
        adjacency=torch.sparse_coo_tensor(index,torch.ones(len(pair)),(4,4)).coalesce()
        graph,support=graph_from_adjacency(adjacency,args.device)
        graphs.append(graph);dense.append(support.to_dense().bool().to(args.device))
    results=[]
    for dtype in (torch.float32,torch.float64):
        sparse=HAN([5,7],3,config).to(device=args.device,dtype=dtype)
        oracle=copy.deepcopy(sparse);sparse.eval();oracle.eval()
        xs=[torch.randn(4,5,device=args.device,dtype=dtype,requires_grad=True),
            torch.randn(4,7,device=args.device,dtype=dtype,requires_grad=True)]
        xd=[x.detach().clone().requires_grad_() for x in xs]
        logits,attention=sparse(xs,graphs,return_attention=True)
        other_logits,other_attention=oracle(xd,dense,dense=True,return_attention=True)
        atol=2e-6 if dtype==torch.float32 else 1e-11
        torch.testing.assert_close(logits,other_logits,atol=atol,rtol=2e-5 if dtype==torch.float32 else 1e-10)
        torch.testing.assert_close(attention,other_attention,atol=atol,rtol=1e-5)
        assert attention.shape==(4,2) and not torch.allclose(attention[0],attention[1])
        torch.testing.assert_close(attention.sum(1),torch.ones(4,device=args.device,dtype=dtype))
        (logits.square().sum()+official_l2_penalty(sparse,.001)).backward()
        (other_logits.square().sum()+official_l2_penalty(oracle,.001)).backward()
        max_grad=0.0
        for (name,p1),(other_name,p2) in zip(sparse.named_parameters(),oracle.named_parameters()):
            assert name==other_name
            torch.testing.assert_close(p1.grad,p2.grad,atol=atol*2,rtol=3e-5 if dtype==torch.float32 else 1e-9)
            max_grad=max(max_grad,(p1.grad-p2.grad).abs().max().item())
        for x1,x2 in zip(xs,xd):torch.testing.assert_close(x1.grad,x2.grad,atol=atol*2,rtol=3e-5)
        results.append({"dtype":str(dtype),"forward_max_error":(logits-other_logits).abs().max().item(),
            "parameter_gradient_max_error":max_grad,"all_input_gradients_checked":True})
    model=HAN([5,7],3,config).to(args.device)
    features=[torch.randn(4,5,device=args.device),torch.randn(4,7,device=args.device)]
    before=[x.clone() for x in features]
    stable=StableInference(model,features,graphs);out=stable(model)
    assert out.dtype==torch.float64 and next(model.parameters()).dtype==torch.float32
    assert all(torch.equal(a,b) for a,b in zip(features,before))
    oracle=copy.deepcopy(model).double().eval()
    expected=oracle([x.double() for x in features],dense,dense=True)
    torch.testing.assert_close(out,expected,atol=1e-11,rtol=1e-10)
    parameter=torch.nn.Parameter(torch.tensor([1.0,-2.0],dtype=torch.float64,device=args.device))
    optimizer=TF1Adam([parameter],lr=.005);expected=parameter.detach().clone()
    m=torch.zeros_like(expected);v=torch.zeros_like(expected)
    for step,gradient in enumerate(([1e-10,-2e-9],[.4,-.3],[.1,.5]),1):
        g=torch.tensor(gradient,dtype=torch.float64,device=args.device)
        parameter.grad=g.clone();optimizer.step();m=.9*m+.1*g;v=.999*v+.001*g.square()
        expected=expected-.005*math.sqrt(1-.999**step)/(1-.9**step)*m/(torch.sqrt(v)+1e-8)
        torch.testing.assert_close(parameter,expected,atol=1e-14,rtol=1e-14)
    copied=TF1Adam([parameter],lr=.1);copied.load_state_dict(optimizer.state_dict())
    assert copied.state[parameter]["step"]==3
    stopping=OfficialEarlyStopping(patience=2)
    assert not stopping.consider(.8,.5)
    assert not stopping.consider(.8,.7) and stopping.curr_step==0  # inclusive ACCtie resets
    assert not stopping.consider(.7,.6) and stopping.curr_step==1
    assert not stopping.consider(.7,.5) and stopping.curr_step==0  # inclusive CEtie resets
    assert not stopping.consider(.6,.8) and stopping.consider(.6,.8)
    record={"status":"passed","device":args.device,"relations":2,"nodes":4,"asymmetric_edges":True,
        "precision_checks":results,"official_per_node_semantic_attention":True,"width":64,"heads":8,
        "stable_inference_independent_fp64_copy":True,"canonical_features_not_mutated":True,
        "tf1_adam_formula_and_state_checked":True,"official_inclusive_early_stop_checked":True,"time":time.time()}
    check_id="smoke_native64_equivalence_"+str(time.time_ns())
    write_json(METHOD_DIR/"exp"/check_id/"checks.json",record)
    log=METHOD_DIR/"log"/check_id;log.mkdir(parents=True)
    (log/"console.log").write_text("HAN_CHECKS_PASSED "+json.dumps(record)+"\n")
    print("HAN_CHECKS_PASSED",json.dumps(record),flush=True)


if __name__=="__main__":main()
