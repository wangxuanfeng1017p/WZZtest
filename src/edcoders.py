# Adapt from: https://github.com/THUDM/GraphMAE/blob/main/graphmae/models/edcoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair

import sympy
import scipy

from utils import obtain_act, obtain_norm, obtain_pooler, sce_loss

# DGL built-in convs (available in DGL 1.0.2)
import dgl.nn.pytorch as dglnn


class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = obtain_act(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(
                in_dim, num_hidden, residual=residual, norm=norm, activation=obtain_act(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(
                    num_hidden, num_hidden, residual=residual, norm=norm, activation=obtain_act(activation)))
            # output projection
            self.gcn_layers.append(GraphConv(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        # if norm is not None:
        #     self.norms = nn.ModuleList([
        #         norm(num_hidden)
        #         for _ in range(num_layers - 1)
        #     ])
        #     if not encoding:
        #         self.norms.append(norm(out_dim))
        # else:
        #     self.norms = None
        self.norms = None
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 activation=None,
                 residual=True,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim

        self.fc = nn.Linear(in_dim, out_dim)

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        # if norm == "batchnorm":
        #     self.norm = nn.BatchNorm1d(out_dim)
        # elif norm == "layernorm":
        #     self.norm = nn.LayerNorm(out_dim)
        # else:
        #     self.norm = None

        self._activation = activation
        if norm == "batchnorm":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == "layernorm":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None

    def forward(self, g, feat):
        """A lightweight GraphConv used by this repo's GCN encoder.

        This used to be a stub (no forward), which breaks GraphMAE pretraining.
        We implement a standard mean-aggregator GCN layer:
        - message: copy_u('h')
        - reduce: mean
        - linear projection + optional residual + optional norm + optional activation
        """

        with g.local_scope():
            h_in = feat
            g.ndata["h"] = feat
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "neigh"))
            h_neigh = g.ndata["neigh"]
            out = self.fc(h_neigh)

            if hasattr(self, "res_fc") and self.res_fc is not None:
                out = out + self.res_fc(h_in)

            if self.norm is not None:
                out = self.norm(out)

            if self._activation is not None:
                out = self._activation(out)

            return out


class GraphSAGE(nn.Module):
    """GraphSAGE encoder using DGL's SAGEConv.

    Notes:
    - We keep the same interface as GCN/GIN: forward(g, x, return_hidden=False)
    - Residual and norm are applied similarly to GCN.
    """

    def __init__(
        self,
        in_dim,
        num_hidden,
        out_dim,
        num_layers,
        dropout,
        activation,
        residual,
        norm,
        encoding=False,
        aggregator_type: str = "mean",
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.aggregator_type = aggregator_type

        last_activation = obtain_act(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(
                dglnn.SAGEConv(
                    in_dim, out_dim, aggregator_type=aggregator_type, feat_drop=0.0, bias=True
                )
            )
            self.proj = nn.ModuleList([nn.Identity()])
        else:
            self.layers.append(
                dglnn.SAGEConv(
                    in_dim, num_hidden, aggregator_type=aggregator_type, feat_drop=0.0, bias=True
                )
            )
            for _ in range(1, num_layers - 1):
                self.layers.append(
                    dglnn.SAGEConv(
                        num_hidden, num_hidden, aggregator_type=aggregator_type, feat_drop=0.0, bias=True
                    )
                )
            self.layers.append(
                dglnn.SAGEConv(
                    num_hidden, out_dim, aggregator_type=aggregator_type, feat_drop=0.0, bias=True
                )
            )
            self.proj = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        # Residual projections
        if residual:
            for i in range(num_layers):
                in_d = in_dim if i == 0 else num_hidden
                out_d = out_dim if i == num_layers - 1 else num_hidden
                if in_d != out_d:
                    self.proj[i] = nn.Linear(in_d, out_d, bias=False)

        self.norms = None
        if norm is not None and encoding:
            self.norms = nn.ModuleList()
            for i in range(num_layers):
                out_d = out_dim if i == num_layers - 1 else num_hidden
                # obtain_norm returns a module class/factory in this repo
                self.norms.append(obtain_norm(norm)(out_d))

        self._last_activation = last_activation
        self._last_norm = last_norm
        self._last_residual = last_residual
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h_in = h
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)

            # residual
            if isinstance(self.proj[l], nn.Identity):
                # Only safe when shapes match
                if h.shape[-1] == h_in.shape[-1]:
                    h = h + h_in
            else:
                h = h + self.proj[l](h_in)

            # activation/norm
            is_last = l == (self.num_layers - 1)
            if not is_last:
                h = obtain_act(self.activation)(h)
                if self.norms is not None:
                    h = self.norms[l](h)
            else:
                if self._last_activation is not None:
                    h = self._last_activation(h)
                if self.norms is not None:
                    h = self.norms[l](h)

            hidden_list.append(h)

        if return_hidden:
            return self.head(h), hidden_list
        return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GAT(nn.Module):
    """GAT encoder using DGL's GATConv.

    We set `num_heads` and keep output dim consistent by using:
        out_dim_per_head = ceil(out_dim / num_heads)
    then project back to out_dim.
    """

    def __init__(
        self,
        in_dim,
        num_hidden,
        out_dim,
        num_layers,
        dropout,
        activation,
        residual,
        norm,
        encoding=False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.num_heads = int(num_heads)

        # choose per-head dims
        def _split_dim(d):
            return int((d + self.num_heads - 1) // self.num_heads)

        self.layers = nn.ModuleList()
        self.proj_out = nn.ModuleList()

        for l in range(num_layers):
            in_d = in_dim if l == 0 else num_hidden
            out_d = out_dim if l == num_layers - 1 else num_hidden
            out_per_head = _split_dim(out_d)
            self.layers.append(
                dglnn.GATConv(
                    in_d,
                    out_per_head,
                    num_heads=self.num_heads,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    residual=False,
                    activation=None,
                    allow_zero_in_degree=True,
                )
            )
            self.proj_out.append(nn.Linear(out_per_head * self.num_heads, out_d, bias=False))

        self.res_proj = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        if residual:
            for l in range(num_layers):
                in_d = in_dim if l == 0 else num_hidden
                out_d = out_dim if l == num_layers - 1 else num_hidden
                if in_d != out_d:
                    self.res_proj[l] = nn.Linear(in_d, out_d, bias=False)

        self.norms = None
        if norm is not None and encoding:
            self.norms = nn.ModuleList()
            for l in range(num_layers):
                out_d = out_dim if l == num_layers - 1 else num_hidden
                self.norms.append(obtain_norm(norm)(out_d))

        self._last_activation = obtain_act(activation) if encoding else None
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h_in = h
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)  # (N, heads, out_per_head)
            h = h.flatten(1)  # (N, heads*out_per_head)
            h = self.proj_out[l](h)

            # residual
            if isinstance(self.res_proj[l], nn.Identity):
                # Only safe when shapes match
                if h.shape[-1] == h_in.shape[-1]:
                    h = h + h_in
            else:
                h = h + self.res_proj[l](h_in)

            # activation/norm
            is_last = l == (self.num_layers - 1)
            if not is_last:
                h = obtain_act(self.activation)(h)
                if self.norms is not None:
                    h = self.norms[l](h)
            else:
                if self._last_activation is not None:
                    h = self._last_activation(h)
                if self.norms is not None:
                    h = self.norms[l](h)

            hidden_list.append(h)

        if return_hidden:
            return self.head(h), hidden_list
        return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)

####################
#     GIN
####################



class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 ):
        super(GIN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = obtain_act(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            apply_func = GIN_MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, norm=norm, activation=activation)
            self.layers.append(GINConv(in_dim, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))
        else:
            # input projection (no residual)
            self.layers.append(GINConv(
                in_dim, 
                num_hidden, 
                ApplyNodeFunc(GIN_MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                init_eps=0,
                learn_eps=learn_eps,
                residual=residual)
                )
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.layers.append(GINConv(
                    num_hidden, num_hidden, 
                    ApplyNodeFunc(GIN_MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                    init_eps=0,
                    learn_eps=learn_eps,
                    residual=residual)
                )
            # output projection
            apply_func = GIN_MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, activation=activation, norm=norm)

            self.layers.append(GINConv(num_hidden, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GINConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 apply_func,
                 aggregator_type="sum",
                 init_eps=0,
                 learn_eps=False,
                 residual=False,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.apply_func = apply_func

        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
            
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            return rst


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        norm_func = obtain_norm(norm)
        if norm_func is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_func(self.mlp.output_dim)
        self.act = obtain_act(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class GIN_MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(GIN_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(obtain_norm(norm)(hidden_dim))
                self.activations.append(obtain_act(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)


####################
#     BWGNN
####################


class PolyConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_dim = in_dim
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_dim, out_feats, bias)
        self.lin = lin
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h



def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_dim, num_hidden, 
                 encoding=False, d=2, dropout=False):
        super(BWGNN, self).__init__()

        # encoder need activation, decode not
        self.is_encoder = encoding
        self.dropout = dropout

        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(num_hidden, num_hidden, self.thetas[i], lin=False)) # always no lin
        self.linear = nn.Linear(in_dim, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        # self.linear3 = nn.Linear(num_hidden*len(self.conv), num_hidden)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0]).to(g.device)
        for conv in self.conv:
            # h0 = conv(g, h)
            h_final = torch.cat([h_final, conv(g, h)], -1)
            # print(h_final.shape)
        # output dim = num_hidden*len(self.conv)
        if self.is_encoder:
            h_final = self.act(h_final)
        # TODO: dropout?
        if self.dropout:
            raise NotImplementedError
        return h_final