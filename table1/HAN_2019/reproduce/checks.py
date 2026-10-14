"""Dense-vs-sparse oracle with directed edges, all relations and gradients."""
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
from reproduce.common import METHOD_DIR,SEARCH
from reproduce.han_model import HAN,graph_from_adjacency,official_l2_penalty
from reproduce.tf1_adam import TF1Adam


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");args=p.parse_args()
    torch.set_num_threads(2);seed_everything(1024)
    config=json.loads(SEARCH.read_text())["candidates"][0]
    # Two genuinely asymmetric relation supports; no undirected oracle shortcut.
    pairs=[[(0,1),(1,2),(2,0),(3,1)],[(0,2),(2,3),(3,0),(1,3)]]
    graphs=[];dense=[]
    for pair in pairs:
        index=torch.tensor(pair).T
        adjacency=torch.sparse_coo_tensor(index,torch.ones(len(pair)),(4,4)).coalesce()
        graph,support=graph_from_adjacency(adjacency,args.device)
        graphs.append(graph);dense.append(support.to_dense().bool().to(args.device))
    sparse=HAN([5,7],3,config).to(args.device);oracle=copy.deepcopy(sparse)
    sparse.eval();oracle.eval()
    xs=[torch.randn(4,5,device=args.device,requires_grad=True),torch.randn(4,7,device=args.device,requires_grad=True)]
    xd=[view.detach().clone().requires_grad_() for view in xs]
    logits,attention=sparse(xs,graphs,return_attention=True)
    dense_logits,dense_attention=oracle(xd,dense,dense=True,return_attention=True)
    torch.testing.assert_close(logits,dense_logits,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(attention,dense_attention,atol=2e-6,rtol=2e-5)
    # Semantic weights remain node-specific and sum over ALL relation channels.
    assert attention.shape==(4,2)
    torch.testing.assert_close(attention.sum(dim=1),torch.ones(4,device=args.device))
    assert not torch.allclose(attention[0],attention[1])
    loss=logits.square().sum()+official_l2_penalty(sparse,0.001)
    dense_loss=dense_logits.square().sum()+official_l2_penalty(oracle,0.001)
    loss.backward();dense_loss.backward()
    max_grad=0.0
    for (name,parameter),(other_name,other) in zip(sparse.named_parameters(),oracle.named_parameters()):
        assert name==other_name
        torch.testing.assert_close(parameter.grad,other.grad,atol=3e-6,rtol=3e-5)
        max_grad=max(max_grad,(parameter.grad-other.grad).abs().max().item())
    for a,b in zip(xs,xd):torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=3e-5)
    # Independent literal TF1 update oracle, including tiny gradients that expose
    # epsilon placement versus torch.optim.Adam. Check optimizer save/restore.
    parameter=torch.nn.Parameter(torch.tensor([1.0,-2.0],dtype=torch.float64,device=args.device))
    optimizer=TF1Adam([parameter],lr=0.005)
    expected=parameter.detach().clone();m=torch.zeros_like(expected);v=torch.zeros_like(expected)
    for step,gradient in enumerate(([1e-10,-2e-9],[.4,-.3],[.1,.5]),1):
        g=torch.tensor(gradient,dtype=torch.float64,device=args.device)
        parameter.grad=g.clone();optimizer.step()
        m=.9*m+.1*g;v=.999*v+.001*g.square()
        expected=expected-0.005*math.sqrt(1-.999**step)/(1-.9**step)*m/(torch.sqrt(v)+1e-8)
        torch.testing.assert_close(parameter,expected,atol=1e-14,rtol=1e-14)
    copied=TF1Adam([parameter],lr=.1);copied.load_state_dict(optimizer.state_dict())
    assert copied.state[parameter]["step"]==3
    record={"status":"passed","device":args.device,"relations":2,"nodes":4,
        "asymmetric_edges":True,"forward_max_error":(logits-dense_logits).abs().max().item(),
        "parameter_gradient_max_error":max_grad,"all_input_gradients_checked":True,
        "official_per_node_semantic_attention":True,"width":256,"heads":8,
        "parameters":sum(p.numel() for p in sparse.parameters()),"tf1_adam_formula_and_state_checked":True,"time":time.time()}
    check_id="smoke_sparse_equivalence_"+str(time.time_ns())
    write_json(METHOD_DIR/"exp"/check_id/"checks.json",record)
    path=METHOD_DIR/"log"/check_id;path.mkdir(parents=True)
    (path/"console.log").write_text("HAN_CHECKS_PASSED "+json.dumps(record)+"\n")
    print("HAN_CHECKS_PASSED",json.dumps(record),flush=True)


if __name__=="__main__":main()
