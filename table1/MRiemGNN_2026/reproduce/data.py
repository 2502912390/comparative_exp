"""Canonical data plus the author's observed composite relation pattern rule."""
import os
import numpy as np
import torch
from shared.coe_benchmark.protocol import json_digest,sha256_file
from .common import METHOD_DIR,tensor_hash,graph_hash

def composite_graphs(indices,n,reverse=True,composite=True):
    """Vectorized exact set/bitwise-OR equivalent of author build_adj_dict.

    Each distinct observed relation combination gets its own native channel.
    No NxN allocation, no sampled edges, and no truncation.
    """
    keys=[];bits=[]
    for view,ix in enumerate(indices):
        r,c=ix.cpu().numpy()
        k=r*np.int64(n)+c
        if reverse:k=np.concatenate([k,c*np.int64(n)+r])
        k=np.unique(k);keys.append(k);bits.append(np.full(len(k),1<<view,dtype=np.int64))
    if not composite:
        items=[(1<<v,k) for v,k in enumerate(keys)]
    else:
        k=np.concatenate(keys);b=np.concatenate(bits)
        order=np.argsort(k,kind='stable');k=k[order];b=b[order]
        first=np.r_[True,k[1:]!=k[:-1]];starts=np.flatnonzero(first)
        unique=k[first];mask=np.bitwise_or.reduceat(b,starts)
        items=[(int(m),unique[mask==m]) for m in np.unique(mask)]
    graphs=[];patterns=[]
    for mask,k in items:
        ix=torch.from_numpy(np.stack([k//n,k%n])).long()
        graph=torch.sparse_coo_tensor(ix,torch.ones(len(k)),(n,n)).coalesce()
        graphs.append(graph);patterns.append(int(mask))
    # The pattern channels encode exactly all original relation supports after
    # the native fixed reverse-edge rule. Empty raw views remain in metadata.
    for v,k in enumerate(keys):
        recovered=[]
        for pattern,g in zip(patterns,graphs):
            if pattern&(1<<v):
                ix=g.indices().numpy();recovered.append(ix[0]*np.int64(n)+ix[1])
        actual=np.unique(np.concatenate(recovered)) if recovered else np.empty(0,dtype=np.int64)
        if not np.array_equal(actual,k):raise ValueError('Composite graph lost a source relation')
    return graphs,patterns

def prepare(data,config,device,transform=None):
    if data.table!=1:raise ValueError('MRiemGNN registered scope is Table1')
    if not all(torch.equal(data.features[0],f) for f in data.features):raise ValueError('Native shared feature input differs between relations')
    raw=[]
    for a in data.provided_adjs:
        a=a.coalesce();ix=a.indices();keep=(a.values()>0)&(ix[0]!=ix[1]);raw.append(ix[:,keep])
    recipe={'data_identity':data.identity(),'reverse':config['add_reverse_edges'],'composite':config['composite_relations'],
            'rule':'native_composite_relation_patterns_v1','adapter_sha256':sha256_file(__file__)}
    key=json_digest(recipe);cache=METHOD_DIR/'exp/_cache';cache.mkdir(parents=True,exist_ok=True)
    file=cache/(data.dataset+'_'+key+'.pt')
    if file.exists():
        saved=torch.load(file,map_location='cpu')
        if saved['recipe']!=recipe:raise ValueError('Input graph cache identity mismatch')
        graphs=saved['graphs'];patterns=saved['patterns']
        if [graph_hash(a) for a in graphs]!=saved['hashes']:raise ValueError('Input graph cache hash mismatch')
    else:
        graphs,patterns=composite_graphs(raw,data.nodes,config['add_reverse_edges'],config['composite_relations'])
        saved={'recipe':recipe,'graphs':graphs,'patterns':patterns,'hashes':[graph_hash(a) for a in graphs]}
        temp=file.with_suffix('.tmp.'+str(os.getpid()));torch.save(saved,temp);os.replace(temp,file)
    x=data.features[0]
    if transform is None:
        train=x[data.indices('train')]
        if config['feature_norm']=='standard':
            transform={'mode':'standard','mean':train.mean(0,keepdim=True),'std':train.std(0,keepdim=True).clamp(min=1e-6)}
        elif config['feature_norm']=='none':transform={'mode':'none'}
        elif config['feature_norm']=='row':transform={'mode':'row'}
        else:raise ValueError('Unsupported native feature norm')
    if transform['mode']!=config['feature_norm']:raise ValueError('Stored feature transform differs from fixed config')
    if transform['mode']=='standard':
        mean,std=transform['mean'].cpu(),transform['std'].cpu()
        if mean.shape!=(1,x.shape[1]) or std.shape!=mean.shape or not torch.isfinite(std).all() or not (std>0).all():raise ValueError('Invalid saved train-only standardizer')
        x=(x-mean)/std
    elif transform['mode']=='row':x=x/x.abs().sum(1,keepdim=True).clamp(min=1e-6)
    meta={'id':'mriemgnn_native_relation_patterns_train_stats_v1','raw_views':len(raw),
          'raw_positive_offdiag_edges':[int(i.shape[1]) for i in raw],
          'native_composite_relations':config['composite_relations'],'native_reverse_edges':config['add_reverse_edges'],
          'patterns':patterns,'channels':len(graphs),'edge_counts':[a._nnz() for a in graphs],
          'derived_graph_hashes':saved['hashes'],'input_cache_key':key,
          'feature_rule':config['feature_norm'],'feature_stats_fit':'train_only',
          'feature_transform_hash':json_digest({k:tensor_hash(v) if torch.is_tensor(v) else v for k,v in transform.items()}),
          'effective_features_hash':tensor_hash(x),'canonical_features_and_provided_graphs_unchanged':True,
          'self_loop_rule':'native loader removes relation diagonals; native learned self-loop branch retained'}
    adjs={}
    for i,a in enumerate(graphs):
        degree=torch.bincount(a.indices()[0],minlength=data.nodes).clamp(min=1).float().unsqueeze(-1)
        adjs[i]=(a.to(device),degree.to(device))
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite feature standardization')
    return x.to(device),adjs,transform,meta
