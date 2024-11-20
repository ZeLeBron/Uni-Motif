# import pandas as pd
# d=pd.read_csv('datasets/ogbg_molpcba/split/scaffold/test.csv.gz',compression='gzip')
# a=pd.read_csv('datasets/ogbg_molpcba/split/scaffold/train.csv.gz',compression='gzip')
import itertools

import seaborn as sn
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import dgl
import pandas as pd
from ogb.graphproppred import PygGraphPropPredDataset
from rdkit import Chem
from torch_geometric.datasets import ZINC
from rdkit.Chem.rdmolops import GetMolFrags
from rdkit.Chem.BRICS import BreakBRICSBonds
from torch_geometric.utils import to_networkx
from graphgps.transform.graph2mol import OGB_Graph_Add_Mol_By_Smiles,ZINC_Graph_Add_Mol
from graphgps.loader.master_loader import join_dataset_splits
from graphgps.transform.gen_subgraph import get_neighbors,is_leaf,extract_subgraphs
from graphgps.transform.bpe import Tokenizer

def load_dataset(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """

    if name == 'ogbg-molhiv':
        dataset = PygGraphPropPredDataset(name=name, root=dataset_dir,pre_transform=OGB_Graph_Add_Mol_By_Smiles('datasets/ogbg_molhiv/mapping/mol.csv.gz'))
    elif name == 'ogbg-molpcba':
        dataset = PygGraphPropPredDataset(name=name, root=dataset_dir,pre_transform=OGB_Graph_Add_Mol_By_Smiles('datasets/ogbg_molpcba/mapping/mol.csv.gz'))
    elif name == 'ZINC':
        dataset = join_dataset_splits(
        [ZINC(root=dataset_dir+'/ZINC', subset=True, split=split,pre_transform=ZINC_Graph_Add_Mol())
        for split in ['train', 'val', 'test']]
    )
    return dataset



name='ZINC'
dataset=load_dataset('datasets',name)
#统计数据集中每个分子的大小和其中环的个数和环的大小
motif_size=np.zeros((10,30),dtype=int)
# print()
ring_num=0
max_ring=0
cut_leafs=False
vocab_path='datasets/ZINC/100_bpe_vocab.txt'
tokenizer = Tokenizer(vocab_path)
for i,data in enumerate(dataset):
    nodes_num=data.num_nodes-10
    # ego
    graph = dgl.DGLGraph((data.edge_index[0], data.edge_index[1]))
    if graph.num_nodes() < data.num_nodes:
        offset = data.num_nodes - graph.num_nodes()
        for i in range(offset):
            graph.add_nodes(1)
    subgraphs_nodes_mask, subgraphs_edges_mask, _ = extract_subgraphs('hop', data.edge_index, data.num_nodes, 3)
    subgraphs_nodes = subgraphs_nodes_mask.nonzero().T
    data.subgraphs_batch = subgraphs_nodes[0]
    for i in range(subgraphs_edges_mask.shape[0]):
        mask = subgraphs_nodes_mask[i]
        nodes = graph.nodes()
        target_nodes = nodes[mask]
        sub_g = dgl.node_subgraph(graph, target_nodes)
        size = sub_g.num_nodes()
        size = min(size, 9)
        motif_size[size][nodes_num]+=1
    # #cut
    # graph = dgl.DGLGraph((data.edge_index[0], data.edge_index[1]))
    # if graph.num_nodes() < data.num_nodes:
    #     offset = data.num_nodes - graph.num_nodes()
    #     for i in range(offset):
    #         graph.add_nodes(1)
    # graph_nx = dgl.to_networkx(graph)
    # graph_nx_undirected = graph_nx.to_undirected()
    # target_g_list = []
    # comp = nx.algorithms.community.girvan_newman(graph_nx_undirected)
    # connected_components = list(nx.algorithms.connected_components(graph_nx_undirected))
    # if len(connected_components) >= 4:
    #     for item in connected_components:
    #         size = min(len(item), 9)
    #         motif_size[size][nodes]+=1    
    # else:
    #     ggg = None
    #     limited = itertools.takewhile(lambda c: len(c) <= 4, comp)
    #     for communities in limited:
    #         ggg = (tuple(sorted(c) for c in communities))
    #     for i in ggg:
    #         size = min(len(i), 9)

    #         motif_size[size][nodes]+=1 
    # #bpe
    # sub=tokenizer(data.mol)
    # for i in sub.nodes:
    #     atom_mapping =  sub.get_node(i).get_atom_mapping()
    #     size=len(atom_mapping.keys())
    #     size=min(size,9)
    #     motif_size[size][nodes]+=1
    # BRICS
    # fragments = GetMolFrags(BreakBRICSBonds(data.mol), asMols = True)
    # for fragment in fragments:
    #     size=fragment.GetNumAtoms()
    #     size=min(size,9)
    #     motif_size[size][nodes]+=1

    # ring
    # rings = nx.cycle_basis(to_networkx(data,to_undirected=True))
    # rings=Chem.GetSymmSSSR(data.mol)
    # for ring in rings:
    #     ring_size=min(len(ring),9)
    #     motif_size[ring_size][nodes]+=1
    # # #ringedge
    # for bond in data.mol.GetBonds():
    #     if not bond.IsInRing():
    #         atom1 = bond.GetBeginAtomIdx()
    #         atom2 = bond.GetEndAtomIdx()
    #         if cut_leafs and (is_leaf(atom1, data) or is_leaf(atom2, data)):
    #             continue
    #         motif_size[2][nodes]+=1

    #ringpath
    # visited = set()
    # for bond in data.mol.GetBonds():
    #     if not bond.IsInRing() and bond.GetIdx() not in visited:
    #         visited.add(bond.GetIdx())
    #         in_path = []
    #         to_do = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    #         while to_do:
    #             next_node = to_do.pop()
    #             in_path.append(next_node)
    #             neighbors = [neighbor for neighbor in get_neighbors(next_node, data) if not is_leaf(neighbor, data) or not cut_leafs]
    #             if not data.mol.GetAtomWithIdx(next_node).IsInRing() and not len(neighbors) > 2:
    #                 #not in ring and not a junction
    #                 new_neighbors = [neighbor for neighbor in neighbors if neighbor not in in_path]
    #                 visited.update([data.mol.GetBondBetweenAtoms(next_node, neighbor).GetIdx() for neighbor in new_neighbors])
    #                 to_do.update(new_neighbors)
    #         path_len=min(len(in_path),9)
    #         motif_size[path_len][nodes]+=1
# print(ring_num,max_nod,max_ring)

# 绘制热力图
df = pd.DataFrame(motif_size)
heatmap=sn.heatmap(df,cmap=plt.get_cmap('Blues'))
cbar=heatmap.collections[0].colorbar
heatmap.invert_yaxis()
cbar.ax.tick_params(labelsize=15)
# heatmap.yaxis.label.set_size(15)
plt.xticks(np.arange(0,30,5),np.arange(10,40,5),fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('datasets/'+name+'_ego3.png',dpi=300)