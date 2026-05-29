import torch
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
import dgl
from torch import nn
from scipy.special import comb
import math
import copy
import numpy as np
from collections import OrderedDict

import itertools
from functools import reduce
import utils

EPS = 1e-5

def apply_edges_distance(edges):
    # L2 norm
    h_edge = torch.linalg.norm(edges.src['h_tmp'] - edges.dst['h_tmp'], dim=1, ord=2)
    return {'h_edge': h_edge}

def SubgraphPooling(h, sg):
    with sg.local_scope():
        sg.ndata['h_tmp'] = h
        sg.update_all(fn.u_mul_e('h_tmp', 'pw', 'm'), fn.sum('m', 'h_tmp'))
        h = sg.ndata['h_tmp'] + h

        return h

class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h

class UNIMLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=3, mlp_layers=2, dropout_rate=0, activation='ReLU', graph_batch_num=1, **kwargs):
        super().__init__()
        # batch size
        self.graph_batch_num = graph_batch_num
        self.num_classes = num_classes
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)

    def forward(self, g, h):
        with g.local_scope():
            num_nodes = h.shape[0]
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            h_hg = torch.cat([h, hg], 0)
            out = self.mlp(h_hg)
            node_logits, graph_logits = torch.split(out, num_nodes, dim=0)
            return node_logits, graph_logits


class UNIMLP_E2E(nn.Module):
    def __init__(
        self,
        in_feats,
        embed_dims=32,
        khop=1,
        activation='ReLU',
        graph_batch_num=1,
        stitch_mlp_layers=1,
        final_mlp_layers=2,
        pretrain_model=None,
        output_route='e',
        input_route='e',
        dropout_rate=0,
        num_classes=2,
        scaling_cross=1.0,
        **kwargs,
    ):
        super().__init__()
        self.graph_batch_num = graph_batch_num
        self.num_classes = num_classes
        self.output_route = [c for c in output_route]
        self.input_route = [c for c in input_route]

        self.act = getattr(nn, activation)() if isinstance(activation, str) else activation
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # encoder from GraphMAE
        self.pretrain_model = pretrain_model

        ######## network structure begin
        # isolated layer 1
        self.layer1 = nn.ModuleDict({k: nn.Sequential() for k in self.input_route})
        for k in self.input_route:
            for _ in range(stitch_mlp_layers):
                self.layer1[k].append(nn.Linear(embed_dims, embed_dims))
                self.layer1[k].append(self.act)

        # agg layer 2
        self.layer2 = nn.ParameterDict({
            ''.join(k): nn.Parameter(data=torch.ones(1), requires_grad=True) if k[0] == k[1]
            else nn.Parameter(data=torch.rand(1) * scaling_cross, requires_grad=True)
            for k in itertools.product(self.output_route, self.input_route)
        })

        # isolated layer 3
        self.layer3 = nn.ModuleDict({k: nn.Sequential() for k in self.input_route})
        for k in self.input_route:
            for _ in range(stitch_mlp_layers):
                self.layer3[k].append(nn.Linear(embed_dims, embed_dims))
                self.layer3[k].append(self.act)

        # agg layer 4
        self.layer4 = nn.ParameterDict({
            ''.join(k): nn.Parameter(data=torch.ones(1), requires_grad=True) if k[0] == k[1]
            else nn.Parameter(data=torch.rand(1) * scaling_cross, requires_grad=True)
            for k in itertools.product(self.output_route, self.input_route)
        })

        # final isolated layer
        self.layer56 = nn.Sequential(self.dropout)
        for _ in range(final_mlp_layers):
            self.layer56.append(nn.Linear(embed_dims, embed_dims))
            self.layer56.append(self.act)
        self.layer56.append(nn.Linear(embed_dims, num_classes))

        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4, self.layer56])
        ######## network structure end

        self.khop = khop
        self.pooling_act = nn.LeakyReLU()
        self.mask_dicts = {}
        self.single_graph = False

    def apply_edges(self, edges):
        return {'h_edge': (edges.src['h'] + edges.dst['h']) / 2}

    def forward(self, g, h, sg_matrix, scen='train', return_emb: bool = False):
        """Forward.

        If return_emb=True, also return a dict of embeddings right before the final classifier.
        Returns:
          - logits_dict (as before)
          - emb_dict (only if return_emb=True)
        """
        if not self.single_graph:
            # deactivate the BN and dropout for encoder
            self.pretrain_model.eval()
            with g.local_scope():
                inner_state = {}
                h = self.pretrain_model.embed(g, h)

                if 'g' in self.output_route:
                    g.ndata['h'] = h
                    inner_state['g'] = dgl.mean_nodes(g, 'h')
                    g.ndata.pop('h')

                if self.khop != 0:
                    h = SubgraphPooling(h, sg_matrix)

                if 'n' in self.output_route:
                    inner_state['n'] = h
                if 'e' in self.output_route:
                    g.ndata['h'] = h
                    g.apply_edges(self.apply_edges)
                    g.ndata.pop('h')
                    inner_state['e'] = g.edata['h_edge']

                emb_dict = {}
                for idx, layer in enumerate(self.layers):
                    if isinstance(layer, nn.ParameterDict):
                        models_last = self.layers[idx - 1]
                        for o_r in self.output_route:
                            inner_state[o_r] = reduce(
                                torch.Tensor.add_,
                                [layer[''.join((o_r, i_r))] * models_last[i_r](inner_state[o_r]) for i_r in self.input_route],
                            )
                    elif idx == 0 or idx == 2:
                        continue
                    else:
                        for o_r in self.output_route:
                            if return_emb and layer is self.layer56 and isinstance(layer, nn.Sequential) and len(layer) >= 2:
                                x = inner_state[o_r]
                                for li in range(len(layer) - 1):
                                    x = layer[li](x)
                                emb_dict[o_r] = x
                                inner_state[o_r] = layer[-1](x)
                            else:
                                inner_state[o_r] = layer(inner_state[o_r])

                if return_emb:
                    return inner_state, emb_dict
                return inner_state

        # single_graph mode
        self.pretrain_model.eval()
        with g.local_scope():
            inner_state = {}
            h = self.pretrain_model.embed(g, h)

            if 'g' in self.output_route:
                g.ndata['h'] = h
                inner_state['g'] = dgl.mean_nodes(g, 'h')
                g.ndata.pop('h')

            if self.khop != 0:
                h = SubgraphPooling(h, sg_matrix)

            if 'n' in self.output_route:
                inner_state['n'] = h[self.mask_dicts['n'][scen], :]
            if 'e' in self.output_route:
                g.ndata['h'] = h
                g.apply_edges(self.apply_edges)
                g.ndata.pop('h')
                inner_state['e'] = g.edata['h_edge'][self.mask_dicts['e'][scen], :]

            emb_dict = {}
            for idx, layer in enumerate(self.layers):
                if isinstance(layer, nn.ParameterDict):
                    models_last = self.layers[idx - 1]
                    for o_r in self.output_route:
                        inner_state[o_r] = reduce(
                            torch.Tensor.add_,
                            [layer[''.join((o_r, i_r))] * models_last[i_r](inner_state[o_r]) for i_r in self.input_route],
                        )
                elif idx == 0 or idx == 2:
                    continue
                else:
                    for o_r in self.output_route:
                        if return_emb and layer is self.layer56 and isinstance(layer, nn.Sequential) and len(layer) >= 2:
                            x = inner_state[o_r]
                            for li in range(len(layer) - 1):
                                x = layer[li](x)
                            emb_dict[o_r] = x
                            inner_state[o_r] = layer[-1](x)
                        else:
                            inner_state[o_r] = layer(inner_state[o_r])

            if return_emb:
                return inner_state, emb_dict
            return inner_state
