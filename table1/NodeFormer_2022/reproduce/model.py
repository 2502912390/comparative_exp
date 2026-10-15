"""Official NodeFormer with bounded edge-loss memory and saved projections.

No attention graph/node truncation. Chunking changes floating-point summation
order only. The original implementation remains untouched in code/nodeformer.py.
"""
import importlib.util
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint

SOURCE = Path(__file__).resolve().parents[1] / 'code/nodeformer.py'
spec = importlib.util.spec_from_file_location('nodeformer_2022_author', SOURCE)
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)


def gumbel_aggregate(qp, kp, value, scaled_gumbels):
    """Rescale each (batch, head, sample) over all keys; factor cancels.

    The unchanged exponential draw can be float32.tiny, giving g/tau=349
    at tau=.25. Exponentiating it directly overflows, although the normalized
    attention is well defined. No clipping, resampling, or sample removal.
    """
    noise = (scaled_gumbels - scaled_gumbels.amax(dim=0, keepdim=True)).exp()
    weighted_keys = kp.unsqueeze(3) * noise.unsqueeze(4)
    num = author.numerator_gumbel(qp, weighted_keys, value)
    den = author.denominator_gumbel(qp, weighted_keys)
    return (num / den.unsqueeze(-1)).mean(3).permute(1, 0, 2, 3)


def stable_kernelized_gumbel(query, key, value, kernel, projection, samples, tau):
    qp = kernel(query / tau**0.5, True, projection).permute(1, 0, 2, 3)
    kp = kernel(key / tau**0.5, False, projection).permute(1, 0, 2, 3)
    value = value.permute(1, 0, 2, 3)
    # Exactly the author's CPU exponential draw and device/temperature order.
    gumbels = (-torch.empty(kp.shape[:-1] + (samples,),
        memory_format=torch.legacy_contiguous_format).exponential_().log()).to(query.device) / tau
    return gumbel_aggregate(qp, kp, value, gumbels)


class EdgeLog(torch.autograd.Function):
    """Exact mean(log(attention_ij)/degree_i) and analytical first gradient."""
    @staticmethod
    def forward(ctx, q, k, den, edge, chunk):
        src, dst = edge
        degree = torch.bincount(dst, minlength=len(q)).to(q.dtype)
        ctx.save_for_backward(q, k, den, edge, degree)
        ctx.chunk = chunk
        total = q.new_zeros(())
        for lo in range(0, len(src), chunk):
            s, t = src[lo:lo+chunk], dst[lo:lo+chunk]
            dot = (q[t] * k[s]).sum(-1)
            total += ((dot / den[t]).log() / degree[t, None]).sum()
        return total / (len(src) * q.shape[1])

    @staticmethod
    def backward(ctx, grad):
        q, k, den, edge, degree = ctx.saved_tensors
        src, dst = edge
        dq, dk, dd = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(den)
        scale = grad / (len(src) * q.shape[1])
        for lo in range(0, len(src), ctx.chunk):
            s, t = src[lo:lo+ctx.chunk], dst[lo:lo+ctx.chunk]
            qt, ks = q[t], k[s]
            factor = scale / (degree[t, None] * (qt * ks).sum(-1))
            dq.index_add_(0, t, ks * factor[..., None])
            dk.index_add_(0, s, qt * factor[..., None])
            dd.index_add_(0, t, -scale / (degree[t, None] * den[t]))
        return dq, dk, dd, None, None


class Conv(author.NodeFormerConv):
    def __init__(self, *args, edge_chunk=16384, activation_checkpoint=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('saved_projection', torch.zeros(self.nb_random_features, self.out_channels))
        self.register_buffer('saved_projection_seed', torch.zeros((), dtype=torch.int64))
        self.freeze_projection = False
        self.edge_chunk = edge_chunk
        self.activation_checkpoint = activation_checkpoint

    def forward(self, z, adjs, tau):
        _, n = z.shape[:2]
        q = self.Wq(z).reshape(-1, n, self.num_heads, self.out_channels)
        k = self.Wk(z).reshape(-1, n, self.num_heads, self.out_channels)
        v = self.Wv(z).reshape(-1, n, self.num_heads, self.out_channels)
        if self.freeze_projection:
            projection = self.saved_projection
        else:
            seed = torch.ceil(torch.abs(torch.sum(q) * author.BIG_CONSTANT)).to(torch.int32)
            projection = author.create_projection_matrix(self.nb_random_features, q.shape[-1], seed=seed).to(q.device)
            self.saved_projection.copy_(projection)
            self.saved_projection_seed.copy_(seed)
        # This detached snapshot must not be overwritten by a later layer/pass.
        projection = projection.detach().clone()
        def aggregate(query, key, value):
            if self.use_gumbel and self.training:
                return stable_kernelized_gumbel(query, key, value,
                    self.kernel_transformation, projection, self.nb_gumbel_sample, tau)
            return author.kernelized_softmax(query, key, value,
                self.kernel_transformation, projection, adjs[0], tau, False)
        if self.training and self.activation_checkpoint:
            out = checkpoint(aggregate, q, k, v, use_reentrant=True, preserve_rng_state=True)
        else:
            out = aggregate(q, k, v)
        for i in range(self.rb_order):
            out = out + author.add_conv_relational_bias(v, adjs[i], self.b[i], self.rb_trans)
        out = self.Wo(out.flatten(-2, -1))
        if not self.use_edge_loss:
            return out
        qp = self.kernel_transformation(q / tau**0.5, True, projection).squeeze(0)
        kp = self.kernel_transformation(k / tau**0.5, False, projection).squeeze(0)
        den = (qp * kp.sum(0)).sum(-1)
        link = EdgeLog.apply(qp, kp, den, adjs[0], self.edge_chunk)
        return out, link


def build_model(in_channels, classes, config):
    keys = ('hidden_channels', 'num_layers', 'num_heads', 'dropout', 'use_bn',
            'use_gumbel', 'use_residual', 'use_act', 'use_jk', 'rb_order', 'rb_trans', 'use_edge_loss')
    kwargs = {k: config[k] for k in keys}
    kwargs.update(nb_random_features=config['M'], nb_gumbel_sample=config['K'])
    model = author.NodeFormer(in_channels, out_channels=classes, **kwargs)
    model.convs = torch.nn.ModuleList([Conv(config['hidden_channels'], config['hidden_channels'],
        config['num_heads'], nb_random_features=config['M'], use_gumbel=config['use_gumbel'],
        nb_gumbel_sample=config['K'], rb_order=config['rb_order'], rb_trans=config['rb_trans'],
        use_edge_loss=config['use_edge_loss'], edge_chunk=config['edge_chunk'],
        activation_checkpoint=config['activation_checkpoint']) for _ in range(config['num_layers'])])
    model.reset_parameters()
    return model


def freeze_projections(model):
    for conv in model.convs:
        conv.freeze_projection = True
