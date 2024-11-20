from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_sub')
def set_cfg_sub(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # example argument group
    cfg.sub = CN()

    # ego subgraph type
    cfg.sub.type = '' #['ring', 'ego','cut', 'brics','bpe','magnet','himp','cut','ringedge']
    cfg.sub.ego_type='hop'  #['hop', 'random']
    cfg.sub.num_hops=2
    cfg.sub.embedding_type='lap_pe' #使用哪种编码ego的位置信息 ['lap_pe', 'rand_walk']
    cfg.sub.egograph_pos_enc_dim=16
    
    # cut subgraph type
    cfg.sub.cut_times=2
    cfg.sub.cut_pos_enc_dim=16

    # ringpath subgraph type
    cfg.sub.vocab_size=8
    cfg.sub.max_ring=0
    cfg.sub.add_fragments=False
    cfg.sub.cut_leafs=False
    # cfg.sub.fragment_emb_dim=16

    cfg.sub.batchnorm = False
    cfg.sub.pooling = 'add' #['add', 'mean', 'max']