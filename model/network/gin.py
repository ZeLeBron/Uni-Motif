import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.config import cfg

from graphgps.encoder.sub_encoder import sub_encoder

@register_network('gin')
class GIN(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()

        self.num_features = cfg.dataset.num_features
        self.num_classes = cfg.dataset.num_classes
        dim = cfg.gnn.dim_inner
        self.dropout = cfg.gnn.dropout

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.x_encoder=nn.Sequential(nn.Linear(self.num_features, dim))
        self.convs.append(GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(self.num_features, self.num_classes))
        self.fcs.append(nn.Linear(dim, self.num_classes))

        if cfg.posenc_RWSE.enable:
            num_rw_steps = len(cfg.posenc_RWSE.kernel.times)
            self.pe_encoder=nn.Sequential(nn.Linear(num_rw_steps, dim), nn.ReLU())
        if cfg.sub.type!='':
            self.sub_encoder=sub_encoder(cfg)
        for _ in range(self.num_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, self.num_classes))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, batch):
        outs = [batch.x]
        x = self.x_encoder(batch.x)
        if cfg.posenc_RWSE.enable:
            x += self.pe_encoder(batch.pestat_RWSE)
        batch.x=x
        # print(cfg.sub.type)
        if cfg.sub.type!=None:
            sub_embedding_pooled=self.sub_encoder(batch)
            x+=sub_embedding_pooled
        edge_index = batch.edge_index
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x)
        
        out = None
        for i, x in enumerate(outs):
            x = global_add_pool(x,  batch.batch)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x
        return out,batch.y