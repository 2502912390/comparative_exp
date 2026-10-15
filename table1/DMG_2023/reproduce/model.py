"""Namespaced extraction of author DMG mechanisms; sparse graph execution only."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Optional, Tuple
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, drop_prob, isBias=False):
        super(GCN, self).__init__()

        self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)
        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)
        self.drop_prob = drop_prob
        self.isBias = isBias


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq = F.dropout(seq, self.drop_prob, training=self.training)
        seq_raw = self.fc_1(seq)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_raw, 0)), 0)
        else:
            seq = torch.sparse.mm(adj, seq_raw) if adj.is_sparse else torch.mm(adj, seq_raw)

        if self.isBias:
            seq += self.bias_1

        return self.act(seq)

class Linearlayer(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Linearlayer, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1*x2))/(sigma1*sigma2)

    return corr

def loss_independence(phi_c_list, psi_p_list):
    # Correlation
    corr = 0
    for i in range(len(phi_c_list)):
        corr += compute_corr(phi_c_list[i], psi_p_list[i])
    return corr

def loss_contrastive(U, private, adj_list, args):
    i = 0
    loss = 0
    for adj in adj_list:
        adj = adj_list[i]
        out_node = adj.to_sparse()._indices()[1]
        random = np.random.randint(out_node.shape[0], size=int((out_node.shape[0] / args.sample_num)))
        sample_edge = adj.to_sparse()._indices().T[random]
        dis = F.cosine_similarity(U[sample_edge.T[0]],U[sample_edge.T[1]])
        a, maxidx = torch.sort(dis, descending=True)
        idx1 = maxidx[:int(sample_edge.shape[0]*0.2)]
        b, minidx = torch.sort(dis, descending=False)
        idx2 = minidx[:int(sample_edge.shape[0]*0.1)]
        private_sample_0 = private[i][sample_edge[idx1].T[0]]
        private_sample_1 = private[i][sample_edge[idx1].T[1]]
        private_sample_2 = private[i][sample_edge[idx2].T[0]]
        private_sample_3 = private[i][sample_edge[idx2].T[1]]
        i += 1
        loss += semi_loss(private_sample_0, private_sample_1, private_sample_2, private_sample_3, args)
    return loss

def semi_loss(z1, z2, z3, z4, args):
    f = lambda x: torch.exp(x / args.tau)
    positive = f(F.cosine_similarity(z1, z2))
    negative = f(F.cosine_similarity(z3, z4))
    return -torch.log(
        positive.sum()
        / (positive.sum() + negative.sum() ))

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradientReversalLayer.apply(x, coeff)

class GNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pipe = GCN(args.ft_size, args.hid_units, args.activation, args.dropout, args.isBias)
        # map to common
        self.S = nn.Linear(args.hid_units, args.c_dim)
        # map to private
        self.P = nn.Linear(args.hid_units, args.p_dim)

    def forward(self, x, adj):
        tmp = self.pipe(x, adj)
        common = self.S(tmp)
        private = self.P(tmp)
        return common, private

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = Linearlayer(args.decolayer,args.c_dim + args.p_dim, args.hid_units, args.ft_size)
        self.linear2 = nn.Linear(args.ft_size, args.ft_size)

    def forward(self, s, p):
        recons = self.linear1(torch.cat((s, p), 1))
        recons = self.linear2(F.relu(recons))
        return recons

class GNNDAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_view = self.args.num_view
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for _ in range(self.args.num_view):
            self.encoder.append(GNNEncoder(args))
            self.decoder.append(Decoder(args))

    def encode(self, x, adj_list):
        common = []
        private = []
        for i in range(self.args.num_view):
            tmp = self.encoder[i](x[i], adj_list[i])
            common.append(tmp[0])
            private.append(tmp[1])

        return common, private

    def decode(self, s, p):
        recons = []
        for i in range(self.num_view):
            tmp = self.decoder[i](s[i], p[i])
            recons.append(tmp)

        return recons

    def forward(self, x, adj):
        common, private = self.encode(x, adj)
        recons = self.decode(common, private)

        return common, private, recons

    def embed(self, x, adj_list):
        common = []
        private = []
        for i in range(self.args.num_view):
            tmp = self.encoder[i](x[i], adj_list[i])
            common.append(tmp[0].detach())
            private.append(tmp[1].detach())
        return common, private

class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]

        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y

class Measure_F(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.psi = MLP(view2_dim, psi_size, latent_dim)
        # gradient reversal layer
        self.grl1 = GradientReversalLayer()
        self.grl2 = GradientReversalLayer()

    def forward(self, x1, x2):
        y1 = self.phi(grad_reverse(x1,1))
        y2 = self.psi(grad_reverse(x2,1))
        return y1, y2

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

@torch.no_grad()
def update_S(model, features, adj_list, c_dim, device=None):
    model.eval()
    common, _ = model.encode(features, adj_list)
    # Official update centers each concatenated view, then sums its blocks.
    # Generalize official hardcoded range(2) to all provided views.
    centered = [x - x.mean(0, keepdim=True) for x in common]
    U, _, T = torch.svd(sum(centered))
    return (U @ T.t()) * (common[0].shape[0] ** 0.5)


def matching_reconstruction(common, recon, features, U, neighbors, args, epoch):
    mse = nn.MSELoss(reduction='sum')
    n = common[0].shape[0]
    matching = mse(torch.cat(common, 1), U.repeat(1, args.num_view)) / n
    feature_reconstruction = sum(mse(r,x) for r,x in zip(recon,features)) / n
    interval = int(args.neighbor_num / args.sample_neighbor)
    topology = 0
    for r,x,neighbor in zip(recon,features,neighbors):
        # Preserve the author's sample_neighbor+1 samples / sample_neighbor denominator.
        total = None
        for j in range(args.sample_neighbor + 1):
            q = (epoch + interval*j) % args.neighbor_num
            target = x[neighbor.at(q)]
            total = target if total is None else total + target
        topology = topology + mse(r, total / args.sample_neighbor)
    return matching, feature_reconstruction, topology/n


def training_step(model, measurements, optimizer, U, features, adjs, neighbors, args, epoch):
    model.train(); measurements.train()
    common, private, recon = model(features, adjs)
    match, reconstruction, topology = matching_reconstruction(common,recon,features,U,neighbors,args,epoch)
    mapped = [m(c,p) for m,c,p in zip(measurements,common,private)]
    correlation = loss_independence([x[0] for x in mapped],[x[1] for x in mapped])
    contrastive = loss_contrastive(U,private,adjs,args)
    loss = match + args.alpha*(reconstruction+topology) - args.beta*correlation + args.lammbda*contrastive
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite native DMG objective; do not alter coefficients or omit terms')
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {k:float(v.detach()) for k,v in dict(loss=loss,matching=match,reconstruction=reconstruction,topology=topology,correlation=correlation,contrastive=contrastive).items()}


def drop_feature_record(x, drop_prob):
    mask=torch.empty((x.size(1),),dtype=torch.float32,device=x.device).uniform_(0,1) < drop_prob
    y=x.clone();y[:,mask]=0
    return y,mask
