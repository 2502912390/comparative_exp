"""Dense/sparse and official function parity without importing author utils/model."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "reproduce"

import argparse,ast,json,types
from pathlib import Path
import torch
import numpy as np
from . import model
from .common import METHOD_DIR,CyclicNeighbors

def official_namespace():
    namespace={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'np':np,'Any':object,'Optional':__import__('typing').Optional,'Tuple':__import__('typing').Tuple}
    for path,names in [(METHOD_DIR/'code/utils/process.py',{'GCN','Linearlayer','update_S','drop_feature'}),(METHOD_DIR/'code/models/DMG.py',{'compute_corr','loss_matching_recons','GradientReversalLayer','grad_reverse','GNNEncoder','Decoder','GNNDAE','MLP','Measure_F','semi_loss','loss_contrastive','loss_independence','trainmultiplex'})]:
        source=path.read_text();tree=ast.parse(source);tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];exec(compile(tree,str(path),'exec'),namespace)
    return namespace

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path);o=p.parse_args();torch.set_num_threads(2);torch.manual_seed(1024)
    official=official_namespace();device=o.device
    n,f=32,7;ix=torch.stack([torch.arange(n).repeat_interleave(n),torch.arange(n).repeat(n)]);weights=torch.linspace(.01,1,n*n);A=torch.sparse_coo_tensor(ix,weights,(n,n)).coalesce().to(device)
    x=torch.randn(n,f,device=device,requires_grad=True);gcn=model.GCN(f,8,'relu',0.).to(device);reference=official['GCN'](f,8,'relu',0.).to(device);reference.load_state_dict(gcn.state_dict())
    y=gcn(x,A);z=reference(x,A.to_dense());torch.testing.assert_close(y,z,atol=1e-5,rtol=1e-5)
    gy=torch.autograd.grad(y.sum(),[x,*gcn.parameters()],retain_graph=True);gz=torch.autograd.grad(z.sum(),[x,*reference.parameters()]);max_gradient_error=max(float((a-b).abs().max()) for a,b in zip(gy,gz))
    for a,b in zip(gy,gz):torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-5)
    args=types.SimpleNamespace(ft_size=f,hid_units=16,c_dim=8,p_dim=2,activation='relu',dropout=0.,isBias=False,num_view=2,decolayer=3,neighbor_num=300,sample_neighbor=30,sample_num=50,tau=.5,alpha=.02,beta=.8,lammbda=3.)
    net=model.GNNDAE(args).to(device);features=[x.detach(),x.detach()+.1];adjs=[A,A];S=model.update_S(net,features,adjs,8);S0=official['update_S'](net,features,adjs,8,device);torch.testing.assert_close(S,S0,atol=1e-4,rtol=1e-4)
    neigh=[CyclicNeighbors(A.cpu().indices(),n).to(device) for _ in adjs]
    for q in (0,1,299):
        expected=torch.tensor([(i*n)+((q+1)%n) for i in range(n)],device=device);torch.testing.assert_close(neigh[0].at(q),A.indices()[1][expected])
    net.train();c,pr,r=net(features,adjs);native=model.matching_reconstruction(c,r,features,S,neigh,args,24);cache=[[a.at(q) for q in range(args.neighbor_num)] for a in neigh];old=official['loss_matching_recons'](c,r,features,S,cache,args,24)
    for a,b in zip(native,(old[0],old[1],old[2])):torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-5)
    # Full objective and gradients use the same np samples, preserves all mechanisms.
    measure=torch.nn.ModuleList([model.Measure_F(8,2,[16,16],[16,16]).to(device) for _ in range(2)])
    optimizer=torch.optim.Adam([{'params':measure.parameters(),'lr':1.},{'params':net.parameters(),'lr':.001}]);np.random.seed(10)
    stats=model.training_step(net,measure,optimizer,S,features,adjs,neigh,args,24)
    args.num_view=3;three=model.GNNDAE(args).to(device);S3=model.update_S(three,[features[0]]*3,[A]*3,8)
    with torch.no_grad():common,_=three.encode([features[0]]*3,[A]*3);center=sum(v-v.mean(0,keepdim=True) for v in common);U,_,T=torch.svd(center);torch.testing.assert_close(S3,U@T.t()*(n**.5))
    report={'status':'passed','device':device,'dense_sparse_asymmetric_forward_gradient':True,'gradient_max_error':max_gradient_error,'two_view_author_S':True,'three_view_all_common_S':True,'cyclic_neighbors_exact':True,'topology_author_off_by_one_preserved':True,'full_objective_backward':stats}
    if o.output:o.output.parent.mkdir(parents=True,exist_ok=True);o.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
