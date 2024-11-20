from copy import deepcopy
import itertools
import numpy as np
import re
import os
from typing import Any, List, Literal
from rdkit import Chem 
from rdkit.Chem.rdmolops import GetMolFrags
from rdkit.Chem.BRICS import BreakBRICSBonds
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops

from torch_geometric.data import Data
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_networkx, subgraph,
                                   to_undirected, to_dense_adj, scatter)
from torch_geometric.utils.num_nodes import maybe_num_nodes                                   
import torch
from torch_sparse import SparseTensor
from scipy import sparse as sp
import networkx as nx
import dgl
from graphgps.transform.magnet import MolDecomposition
from graphgps.transform.bpe import Tokenizer
fragment2type = {"ring": 0 , "path": 1, "junction": 2}
ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Cu", "Zn", 'Co', "Mn", 'As', 'Al', 'Ni', 'Se', 'Si', 'H', 'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Fe', 'Ga', 'Ge', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']

class SubgraphsData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        self.g = None
        if bool(re.search('(fragments_edge_index)', key)):
            return torch.tensor([[self.x.size(0)], [self.fragments.size(0)]])
        elif bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')]+'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            # should use number of subgraphs or number of supernodes.
            return 1+getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)|(selected_supernodes)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            # batched_edge_attr[subgraphs_edges_mapper] shoud be batched_combined_subgraphs_edge_attr
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
class generate_subgraph(object):
    def __init__(self, cfg):
        self.sub_type = cfg.sub.type
     
        if self.sub_type and self.sub_type not in ['ring', 'ego','cut', 'brics','bpe','ringedge','ringpath']:
            raise ValueError(f"Unexpected PE stats selection {self.sub_type } in {self.sub_type}")
        self.sub_type=cfg.sub.type
        self.num_hops = cfg.sub.num_hops
        self.ego_type = cfg.sub.ego_type
        self.embedding_type = cfg.sub.embedding_type
        self.egograph_pos_enc_dim = cfg.sub.egograph_pos_enc_dim
        self.cut_times = cfg.sub.cut_times
        self.cut_pos_enc_dim = cfg.sub.cut_pos_enc_dim
        self.vocab_size = cfg.sub.vocab_size
        self.max_ring = cfg.sub.max_ring
        self.cut_leafs = cfg.sub.cut_leafs
        self.format = cfg.dataset.format
        if self.sub_type == 'bpe':
            if self.format.startswith('PyG-'):
                self.dataset_id=self.format.split('-',1)[1]
            elif self.format=='OGB':
                self.dataset_id=cfg.dataset.name.replace('-','_')
            else:
                raise ValueError(f"Unexpected dataset format {self.format}")
            self.vocab_path='datasets/{}/{}_bpe_vocab.txt'.format(self.dataset_id,self.vocab_size)
            if os.path.exists(self.vocab_path):
                self.tokenizer=Tokenizer(self.vocab_path)
                self.vocab_size=len(self.tokenizer.idx2subgraph)
            else:
                raise ValueError(f"Vocab file {self.vocab_path} not found")
        

    def __call__(self, data):
        
        if self.sub_type=='ego':
            data=ego_sub(data, self.num_hops, self.ego_type, self.embedding_type, self.egograph_pos_enc_dim)
        elif self.sub_type=='cut':
            data=cut_sub(data, self.cut_times, self.embedding_type, self.cut_pos_enc_dim)
        elif self.sub_type=='ring':
            data=ring_sub(data, vocab_size=self.vocab_size)
            data=gen_representation(data, vocab_size=self.vocab_size)
        elif self.sub_type=='ringedge':
            data=ring_sub(data, vocab_size=self.vocab_size-1)
            data=ring_edge_sub(data, vocab_size=self.vocab_size, cut_leafs = self.cut_leafs)
            data=gen_representation(data, vocab_size=self.vocab_size)
        elif self.sub_type=='magnet':
            data=magnet_sub(data)
            data=gen_representation(data)
        elif self.sub_type=='brics':
            data=brics_sub(data)
            data=gen_representation(data)
        elif self.sub_type=='bpe':
            data=bpe_sub(data,self.tokenizer)
            data=gen_representation(data,vocab_size=self.vocab_size)
        elif self.sub_type=='ringpath':
            data=ring_sub(data, vocab_size=self.max_ring)
            data=ringpath_sub(data, max_ring=self.max_ring,vocab_size=self.vocab_size,cut_leafs = self.cut_leafs)
            data=gen_representation(data, vocab_size=self.vocab_size)
        data = SubgraphsData(**{k: v for k, v in data})
        return data

def get_frag_type(self, type: Literal["ring", "path"], size,max_ring=6,max_path=4):
    if type == "ring":
        return size - 3 if size - 3  < max_ring else max_ring - 1
    else: # type == "path"
        offset = max_ring
        return  offset + size - 2 if size - 2 < max_path else offset + max_path - 1

def ringpath_sub(graph, max_ring=6,vocab_size=10,cut_leafs = False):
    max_path=vocab_size-max_ring
    #now find paths
    max_frag_id = max([frag_id for frag_infos in graph.substructures for (frag_id, _) in frag_infos], default = -1)
    fragment_id = max_frag_id + 1

    fragment_types = []

    #find paths
    visited = set()
    for bond in graph.mol.GetBonds():
        
        if not bond.IsInRing() and bond.GetIdx() not in visited:
            if cut_leafs and is_leaf(bond.GetBeginAtomIdx(), graph) and is_leaf(bond.GetEndAtomIdx(), graph):
                continue
            visited.add(bond.GetIdx())
            in_path = []
            to_do = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            while to_do:
                next_node = to_do.pop()
                in_path.append(next_node)
                neighbors = [neighbor for neighbor in get_neighbors(next_node, graph) if not is_leaf(neighbor, graph) or not cut_leafs]
                if not graph.mol.GetAtomWithIdx(next_node).IsInRing() and not len(neighbors) > 2:
                    #not in ring and not a junction
                    new_neighbors = [neighbor for neighbor in neighbors if neighbor not in in_path]
                    visited.update([graph.mol.GetBondBetweenAtoms(next_node, neighbor).GetIdx() for neighbor in new_neighbors])
                    to_do.update(new_neighbors)
            
            path_info = (fragment_id, get_frag_type("path", len(in_path),max_ring,max_path))
            fragment_types.append([fragment2type["path"], len(in_path)])
            fragment_id += 1
            for node_id in in_path:
                graph.substructures[node_id].append(path_info)
    graph.fragment_types = torch.concat([graph.fragment_types, torch.tensor(fragment_types, dtype = torch.long)], dim = 0)
    return graph


def bpe_sub(graph,tokenizer):
    subgraph_mol = tokenizer(graph.mol)
    node_substructures = [[] for _ in range(graph.x.size(0))]
    for fragment_id,fragment in enumerate(subgraph_mol.nodes):
        atom_mapping =  subgraph_mol.get_node(fragment).get_atom_mapping()
        atom_ids = list(atom_mapping.keys())
        tmp=subgraph_mol.get_node(fragment).smiles
        if tmp in tokenizer.subgraph2idx:
            fragment_type = tokenizer.subgraph2idx[tmp]
        else:  
            fragment_type = -1
        for atom in atom_ids:
            node_substructures[atom].append((fragment_id, fragment_type))
    graph.substructures = node_substructures
    return graph

def brics_sub(graph):
    mol = graph.mol
    node_substructures = [[] for _ in range(graph.num_nodes)]

    fragments = GetMolFrags(BreakBRICSBonds(mol), asMols = True)
    fragments_atom_ids = GetMolFrags(BreakBRICSBonds(mol))

    fragment_id = 0
    fragment_type = None
    for _ , atom_ids in zip(fragments, fragments_atom_ids):
        #filter atom ids that are not introduced by BRICS
        atom_ids_filtered = [atom_id for atom_id in atom_ids if atom_id < graph.num_nodes]

        for id in atom_ids_filtered:
            node_substructures[id].append((fragment_id, fragment_type))

        fragment_id += 1
            
    graph.substructures = node_substructures
    return graph



def magnet_sub(graph):
    mols = Chem.Mol(graph.mol) #create copy of molecule
    fragment_types = []
    node_substructures = []
    for mol in Chem.rdmolops.GetMolFrags(mols, asMols = True):
        #There can be multiple disconnected parts of a molecule
        decomposition = MolDecomposition(mol)
                    
        fragment_to_index = {}
        fragment_to_type = {}
        for fragments in decomposition.nodes.values():
            fragment_info = []
            for frag_id in fragments:
                if frag_id == -1:
                    #don't use leafs ?!
                    continue

                if frag_id not in fragment_to_type:
                    frag_mol = Chem.MolFromSmiles(decomposition.id_to_fragment[frag_id])
                    if frag_mol.GetAtomWithIdx(0).IsInRing():
                        fragment_to_type[frag_id] = [fragment2type["ring"], frag_mol.GetNumAtoms()]
                    elif all([a.GetDegree() in [1, 2] for a in frag_mol.GetAtoms()]):
                        fragment_to_type[frag_id] = [fragment2type["path"],frag_mol.GetNumAtoms()]
                    else:
                        fragment_to_type[frag_id] = [fragment2type["junction"], frag_mol.GetNumAtoms()]

                if frag_id not in fragment_to_index:
                    fragment_to_index[frag_id] = len(fragment_types)
                    fragment_types.append(fragment_to_type[frag_id])

                fragment_info.append((fragment_to_index[frag_id], fragment_to_type[frag_id][0]))

            node_substructures.append(fragment_info)
    
    if fragment_types:
        graph.fragment_types = torch.tensor(fragment_types, dtype = torch.long)
    else:
        graph.fragment_types = torch.empty((0,2), dtype = torch.long)
    graph.substructures = node_substructures
    return graph

def ring_edge_sub(graph, vocab_size, cut_leafs = False):
    #now find edges not in rings
    max_ring=vocab_size-1
    max_frag_id = max([frag_id for frag_infos in graph.substructures for (frag_id, _) in frag_infos], default = -1)
    fragment_id = max_frag_id + 1

    fragment_types = []

    for bond in graph.mol.GetBonds():
        if not bond.IsInRing():
            #add bond as new fragment
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()
            if cut_leafs and (is_leaf(atom1, graph) or is_leaf(atom2, graph)):
                continue
            fragment_types.append([fragment2type["path"],2])
            bond_info = (fragment_id, max_ring)
            fragment_id += 1
            graph.substructures[atom1].append(bond_info)
            graph.substructures[atom2].append(bond_info)
    
    graph.fragment_types = torch.concat([graph.fragment_types, torch.tensor(fragment_types, dtype = torch.long)], dim = 0)
    return graph


def get_neighbors(node_id, graph):
    return (graph.edge_index[1, graph.edge_index[0,:] == node_id]).tolist()

def get_degree(node_id, graph):
    return len(get_neighbors(node_id, graph))

def is_leaf(node_id, graph):
    neighbors = get_neighbors(node_id, graph)
    if len(neighbors) == 1:
        neighbor = neighbors[0]
        if graph.mol.GetAtomWithIdx(neighbor).IsInRing():
            return True
        nns = get_neighbors(neighbor, graph)
        degree_nn = [get_degree(nn, graph) for nn in nns]
        if len([degree for degree in degree_nn if degree >= 2]) >= 2:
            return True
        # one neighbor neighbor with degree one is not a leaf
        potential_leafs = [nn for nn in nns if get_degree(nn, graph) == 1]
        atom_types = [(ATOM_LIST.index(graph.mol.GetAtomWithIdx(nn).GetSymbol()), nn) for nn in potential_leafs]
        sorted_idx = np.sort(atom_types)
        if sorted_idx[-1][1] == node_id:
            #node at end of path
            return False
        else:
            return True
    return False
        

def ring_sub(data,vocab_size):
    # graph = dgl.DGLGraph((data.edge_index[0], data.edge_index[1]))
    # if graph.num_nodes() < data.num_nodes:
    #     offset = data.num_nodes - graph.num_nodes()
    #     for i in range(offset):
    #         graph.add_nodes(1)
    # graph_nx = dgl.to_networkx(graph)
    # graph_nx_undirected = graph_nx.to_undirected()
    max_ring=vocab_size+2

    rings = nx.cycle_basis(to_networkx(data,to_undirected=True))
    node_substructures = [[] for _ in range(data.num_nodes)]
    fragment_types = []
    fragment_id = 0
    rings_set=set()
    for i in range(len(rings)):
        if len(rings[i]) < 3:
            continue
        ring = rings[i]
        rings_set.update(ring)
        fragment_types.append([fragment2type["ring"], len(ring)])
        if len(ring) <= max_ring:
            for atom in ring:
                fragment_type = len(ring) - 3
                node_substructures[atom].append((fragment_id, fragment_type))
            fragment_id += 1
        else:
            for atom in ring:
                fragment_type = vocab_size - 1 # max fragment_type number
                node_substructures[atom].append((fragment_id, fragment_type))
            fragment_id += 1
    data.rings = list(rings_set)
    data.substructures = node_substructures
    if fragment_types:
        data.fragment_types = torch.tensor(fragment_types, dtype = torch.long)
    else:
        data.fragment_types = torch.empty((0,2), dtype = torch.long)
    return data

def gen_representation(data,vocab_size=None):
    #统计该分子图中的fragments字典，key为frag_id，value为该Fragment的类型
    frag_id_to_type = dict([frag_info for frag_infos in data.substructures for frag_info in frag_infos if frag_info])
    #max_frag_id最大的id号，该分子图中所有的framents的个数为max_frag_id+1
    max_frag_id = max([frag_id for frag_infos in data.substructures for (frag_id, _) in frag_infos], default = -1)
    #为每个frag_id创建一个one-hot向量，frag_representation的shape为(max_frag_id+1, vocab_size)
    if vocab_size is not None:
        frag_representation = torch.zeros(max_frag_id +1, vocab_size, dtype=torch.int64)
        frag_representation[list(range(max_frag_id +1)), [frag_id_to_type[frag_id] for frag_id in range(max_frag_id +1)]] = 1
        data.fragments = frag_representation
    else:
        data.fragments = torch.zeros(max_frag_id +1, 0)
    edges = [[node_id, frag_id] for node_id, frag_infos in enumerate(data.substructures) for (frag_id, _) in frag_infos]
    if not edges:
        data.fragments_edge_index = torch.empty((2,0), dtype = torch.long)
    else:
        data.fragments_edge_index = torch.tensor(edges, dtype = torch.long).T.contiguous()
    return data
  
def get_substructure_edge_index(substructure):
    """Compute node-to-substructure edge index.

    Parameters
    ----------
    substructure
        List of substructure tuples with node ids.

    Returns
    -------
        Pytorch-geometric style edge index from nodes to substructures in which the nodes are part of.
    """
    if not substructure:
        return torch.empty(size = (2,0), dtype = torch.long)
    return torch.tensor([[node_id, sub_id]  for sub_id, sub in enumerate(substructure) for node_id in sub], dtype = torch.long).t().contiguous()

def ego_sub(data, num_hops, ego_type, embedding_type, egograph_pos_enc_dim):
    graph = dgl.DGLGraph((data.edge_index[0], data.edge_index[1]))
    if graph.num_nodes() < data.num_nodes:
        offset = data.num_nodes - graph.num_nodes()
        for i in range(offset):
            graph.add_nodes(1)
    subgraphs_nodes_mask, subgraphs_edges_mask, _ = extract_subgraphs(ego_type, data.edge_index, data.num_nodes, num_hops)
    # subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense)
    subgraphs_nodes = subgraphs_nodes_mask.nonzero().T
    data.subgraphs_batch = subgraphs_nodes[0]
    # data.subgraphs_nodes_mapper = subgraphs_nodes[1]
    # data.subgraphs_edges_mapper = subgraphs_edges[1]
    Ego_RWPE = []
    for i in range(subgraphs_edges_mask.shape[0]):
        mask = subgraphs_nodes_mask[i]
        nodes = graph.nodes()
        target_nodes = nodes[mask]
        sub_g = dgl.node_subgraph(graph, target_nodes)
        if embedding_type == 'lap_pe':
            ego_rwpe = lap_positional_encoding(sub_g, egograph_pos_enc_dim)
        else: # 'rand_walk'
            ego_rwpe = init_positional_encoding(sub_g, egograph_pos_enc_dim)
        Ego_RWPE.append(ego_rwpe)
    Ego_RWPE = torch.cat(Ego_RWPE, 0)
    data.sub_pe= Ego_RWPE
    return data

def cut_sub(data, cut_times, embedding_type, cut_pos_enc_dim):
    graph = dgl.DGLGraph((data.edge_index[0], data.edge_index[1]))
    if graph.num_nodes() < data.num_nodes:
        offset = data.num_nodes - graph.num_nodes()
        for i in range(offset):
            graph.add_nodes(1)
    graph_nx = dgl.to_networkx(graph)
    graph_nx_undirected = graph_nx.to_undirected()
    target_g_list = []
    comp = nx.algorithms.community.girvan_newman(graph_nx_undirected)
    connected_components = list(nx.algorithms.connected_components(graph_nx_undirected))
    if len(connected_components) >= cut_times:
        for item in connected_components:
            target_g_list.append(graph_nx.subgraph(item))
    else:
        ggg = None
        limited = itertools.takewhile(lambda c: len(c) <= cut_times, comp)
        for communities in limited:
            ggg = (tuple(sorted(c) for c in communities))
        for i in ggg:
            target_g_list.append(graph_nx.subgraph(i))
    Cut_RWPE = []
    subgraph_x_index = []

    for g in target_g_list:
        subgraph_x_index.append(torch.tensor(list(g.nodes)))
        g_dgl = dgl.from_networkx(g)
        if embedding_type == 'lap_pe':
            cut_rwpe = lap_positional_encoding(g_dgl, cut_pos_enc_dim)
        else: #random_walk
            cut_rwpe = init_positional_encoding(g_dgl, cut_pos_enc_dim)
        Cut_RWPE.append(cut_rwpe)
    cut_RWPE = torch.cat(Cut_RWPE, dim=0)
    data.subgraph_x_index = torch.cat(subgraph_x_index, dim=-1)
    data.sub_pe = cut_RWPE
    return data


def lap_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N  # 得到拉普拉斯矩阵

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())  # 获得特征值、特征向量
    idx = EigVal.argsort()  # increasing order #从小到大排列特征值
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])  # 让特征值和特征向量按照特征值从小到大顺序一一对应

    pos_enc_emb = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    if pos_enc_emb.shape[-1] < pos_enc_dim:
        offset = pos_enc_dim - pos_enc_emb.shape[-1]
        pos_enc_emb = torch.cat((pos_enc_emb, torch.zeros(pos_enc_emb.shape[0], offset)), dim=-1)

    return pos_enc_emb


def init_positional_encoding(g, pos_enc_dim, type_init='rand_walk'):
    """
        Initializing positional encoding with RWPE
    """

    n = g.number_of_nodes()

    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
        RW = A * Dinv
        M = RW  # 随机游走一次后的子图

        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc - 1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE, dim=-1)
        g.ndata['pos_enc'] = PE

    return PE

def to_sparse(node_mask, edge_mask, hop_indicator):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    if hop_indicator is not None:
        hop_indicator = hop_indicator[subgraphs_nodes[0], subgraphs_nodes[1]]
    return subgraphs_nodes, subgraphs_edges, hop_indicator

def extract_subgraphs(ego_type, edge_index, num_nodes, num_hops, sparse=False):
    if ego_type == 'hop':
        node_mask, hop_indicator = k_hop_subgraph(edge_index, num_nodes, num_hops)
    else :
        node_mask, hop_indicator = random_walk_subgraph(edge_index, num_nodes, num_hops, cal_hops=True)
    edge_mask = node_mask[:, edge_index[0]] & node_mask[:, edge_index[1]] # N x E dense mask matrix
    if not sparse:
        return node_mask, edge_mask, hop_indicator
    else:
        return to_sparse(node_mask, edge_mask, hop_indicator)
    
def k_hop_subgraph(edge_index, num_nodes, num_hops):
    # return k-hop subgraphs for all nodes in the graph
    row, col = edge_index
    sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)] # each one contains <= i hop masks  # 返回斜线为1的二维张量
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)   #填充全部为-1 形状为num_nodes * num_nodes的矩阵
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator==-1) & next_mask] = i+1
    hop_indicator = hop_indicator.T  # N x N 
    node_mask = (hop_indicator >= 0) # N x N dense mask matrix
    return node_mask, hop_indicator

from torch_cluster import random_walk
def random_walk_subgraph(edge_index, num_nodes, walk_length, p=1, q=1, repeat=1, cal_hops=True, max_hops=10):
    """
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)  Setting it to a high value (> max(q, 1)) ensures 
            that we are less likely to sample an already visited node in the following two steps.
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
            if q > 1, the random walk is biased towards nodes close to node t.
            if q < 1, the walk is more inclined to visit nodes which are further away from the node t.
        p, q ∈ {0.25, 0.50, 1, 2, 4}.
        Typical values:
        Fix p and tune q 

        repeat: restart the random walk many times and combine together for the result

    """
    row, col = edge_index
    start = torch.arange(num_nodes, device=edge_index.device)
    walks = [random_walk(row, col, 
                         start=start, 
                         walk_length=walk_length,
                         p=p, q=q,
                         num_nodes=num_nodes) for _ in range(repeat)]
    walk = torch.cat(walks, dim=-1)
    node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
    # print(walk.shape)
    node_mask.fill_(False)
    node_mask[start.repeat_interleave((walk_length+1)*repeat), walk.reshape(-1)] = True
    if cal_hops: # this is fast enough
        sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)]
        hop_indicator = row.new_full((num_nodes, num_nodes), -1)
        hop_indicator[hop_masks[0]] = 0
        for i in range(max_hops):
            next_mask = sparse_adj.matmul(hop_masks[i].float())>0
            hop_masks.append(next_mask)
            hop_indicator[(hop_indicator==-1) & next_mask] = i+1
            if hop_indicator[node_mask].min() != -1:
                break 
        return node_mask, hop_indicator
    return node_mask, None

if __name__ == '__main__':
    from rdkit import Chem                          # 引入化学信息学的包rdkit
    from rdkit.Chem import GetAdjacencyMatrix       # 构建分子邻接矩阵
    from scipy.sparse import coo_matrix             # 转化成COO格式 
    import torch
    from torch_geometric.data import Data           # 引入pyg的Data
    import numpy as np
    smiles='O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C'
    mol=Chem.MolFromSmiles(smiles)                  # 从smiles字符串中构建分子
    adj=GetAdjacencyMatrix(mol)                     # 构建分子邻接矩阵
    adj=coo_matrix(adj)                             # 转化成COO格式
    edge_index=[adj.row,adj.col]                    # 构建边索引
    x=np.random.randn(adj.shape[0],10)              # 构建节点特征
    data=Data(x=torch.tensor(x),edge_index=torch.tensor(edge_index),mol=mol)  # 构建pyg的Data
    tokenizer=Tokenizer('datasets/ZINC/300_bpe_vocab.txt')
    data=bpe_sub(data,tokenizer)  # 构建环状子图
    print(data)

