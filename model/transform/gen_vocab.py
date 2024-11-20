import pickle
from rdkit import Chem
import os
from torch_geometric.datasets import ZINC
from ogb.graphproppred import PygGraphPropPredDataset
from graphgps.transform.bpe import graph_bpe_smiles


class PrincipalSubgraphVocab(object):
    def __init__(self, vocab_size = 200, vocab_path = "./principal_subgraph_vocab.txt", cpus = 4, kekulize = False):
        self.max_vocab_size = vocab_size
        self.smis = []
        self.vocab_path = vocab_path
        self.cpus = cpus
        self.kekulize = kekulize
    
    def __call__(self, graph):
        self.smis.append(Chem.MolToSmiles(graph.mol))
        return graph

    def get_vocab(self):
        graph_bpe_smiles(self.smis, vocab_len = self.max_vocab_size, vocab_path = self.vocab_path, cpus = self.cpus, kekulize = self.kekulize)
        
def gen_vocab(dataset_id,vocab_size = 300, cpus = 4, kekulize = False):
    vocab_path='datasets/{}/{}_bpe_vocab.txt'.format(dataset_id.replace('-','_'),vocab_size)
    if os.path.exists(vocab_path):
        print ('Vocab already exists:',vocab_path)
        return
    if dataset_id == 'ZINC':
        data=ZINC(root='datasets/ZINC',subset=True,split='train')
    elif dataset_id == 'ogbg-molhiv':
        data=PygGraphPropPredDataset(name='ogbg-molhiv',root='datasets')
        split_idx=data.get_idx_split()
        data=data[split_idx['train']]
    elif dataset_id == 'ogbg-molpcba':
        data=PygGraphPropPredDataset(name='ogbg-molpcba',root='datasets')
        split_idx=data.get_idx_split()
        data=data[split_idx['train']]
    else:
        raise ValueError('Dataset not supported')
    gen=PrincipalSubgraphVocab(vocab_size = vocab_size, vocab_path = vocab_path, cpus = cpus, kekulize = kekulize)
    for d in data:
        gen(d)
    gen.get_vocab()
    
if __name__ == "__main__":
    for vocab_size in [100,300,500]:
        gen_vocab('ZINC',vocab_size)
        gen_vocab('ogbg-molhiv',vocab_size)
        gen_vocab('ogbg-molpcba',vocab_size)
#     atomic_numbers_to_symbols = {
#     1: 'H',  2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
#     9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S',
#     17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr',
#     25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge',
#     33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
#     41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd',
#     49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba',
#     57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
#     65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf',
#     73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
#     81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra',
#     89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm',
#     97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr',
#     104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds',
#     111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts',
#     118: 'Og'
# }


#     #统计hiv数据集中各类元素的出现次数到一个字典中
#     data=ZINC(root='datasets/ZINC',subset=True,split='train')
#     # split_idx=data.get_idx_split()
#     # data=data[split_idx['train']]
#     atom_count={}
#     for d in data:
#         num_nodes = d.num_nodes
#         for i in range(num_nodes):
#             atom = d.x[i,0].item()+1
#             symbol=atomic_numbers_to_symbols.get(atom,'Unknown')
#             if symbol == 'Unknown':
#                 print('Unknown atom:',atom)
#             if symbol not in atom_count:
#                 atom_count[symbol]=0
#             atom_count[symbol]+=1
#     print(atom_count)