import torch
import torch.nn as nn
from torch_scatter import scatter

from torch_geometric.graphgym.register import (
    register_node_encoder,
)

@register_node_encoder('sub')
class sub_encoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sub_encoder=None
        self.sub_type=cfg.sub.type
        self.bn=cfg.sub.batchnorm
        self.hdim=cfg.gnn.dim_inner
        self.pooling=cfg.sub.pooling
        if self.sub_type == 'ego':
            self.sub_encoder=nn.Linear(cfg.sub.egograph_pos_enc_dim, self.hdim)
        elif self.sub_type == 'cut':
            self.sub_encoder=nn.Linear(cfg.sub.cut_pos_enc_dim, self.hdim)
        elif self.sub_type in ['ring', 'ringedge','ringpath', 'magnet', 'brics','bpe']:
            #待完成！！！！！
            self.add_fragments=cfg.sub.add_fragments
            layers=[]
            if self.add_fragments:
                self.fragment_emb = nn.Embedding(cfg.sub.vocab_size, self.hdim)
                self.fragments_encoder=nn.Linear(self.hdim, self.hdim)
                layers.append(nn.Linear(self.hdim*2, self.hdim))
            else:
                layers.append(nn.Linear(self.hdim, self.hdim))
            # layers.append(nn.ReLU())
            self.sub_encoder=nn.Sequential(*layers)
    def forward(self, batch):
        if self.sub_type=='ego':
            if self.bn == True:
                sub_norm = nn.BatchNorm1d(self.egograph_pos_enc_dim)
                batch.sub_pe = sub_norm(batch.sub_pe)
            sub_embedding = self.sub_encoder(batch.sub_pe)
            sub_embedding_pooled= scatter(sub_embedding, batch.subgraphs_batch, dim=0, reduce=self.pooling)
        elif self.sub_type=='cut':
            cut_sub_embedding = self.sub_encoder(batch.sub_pe)
            cut_sub_embedding_trans = torch.zeros_like(cut_sub_embedding).to(cut_sub_embedding)
            cut_sub_embedding_trans[batch.subgraph_x_index] = cut_sub_embedding
            sub_embedding_pooled= cut_sub_embedding_trans
        elif self.sub_type in ['ring', 'ringedge','ringpath', 'brics','bpe']:
            row, col = batch.fragments_edge_index
            #m V->f
            substructure_x = scatter(batch.x[row], col, reduce = self.pooling, dim = 0)
            if self.add_fragments:
                frag_tmp=torch.argmax(batch.fragments,dim=1)
                fragment_emb=self.fragment_emb(frag_tmp)
                fragments_emb=self.fragments_encoder(fragment_emb)
                substructure_x=torch.cat([substructure_x, fragments_emb], dim=1)
            fragment_emb=self.sub_encoder(substructure_x)
            sub_embedding_pooled=scatter(fragment_emb[col], row, reduce = self.pooling, dim = 0,dim_size=batch.num_nodes)
            # 这里变换后要不要再加一层网络？？？？？
        else:
            raise ValueError(f"Unknown subgraph type: {self.sub_type}")
        return sub_embedding_pooled