#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import networkx as nx
from copy import copy, deepcopy
import numpy as np
from typing import Union

from copy import copy
import argparse
import multiprocessing as mp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDKitMol
from rdkit.Chem.Draw import rdMolDraw2D

class SubgraphNode:
    '''
    The node representing a subgraph
    '''
    def __init__(self, smiles: str, pos: int, atom_mapping: dict, kekulize: bool):
        self.smiles = smiles
        self.pos = pos
        self.mol = smi2mol(smiles, kekulize, sanitize=False)
        # map atom idx in the molecule to atom idx in the subgraph (submol)
        self.atom_mapping = copy(atom_mapping)
    
    def get_mol(self):
        '''return molecule in rdkit form'''
        return self.mol

    def get_atom_mapping(self):
        return copy(self.atom_mapping)

    def __str__(self):
        return f'''
                    smiles: {self.smiles},
                    position: {self.pos},
                    atom map: {self.atom_mapping}
                '''


class SubgraphEdge:
    '''
    Edges between two subgraphs
    '''
    def __init__(self, src: int, dst: int, edges: list):
        self.edges = copy(edges)  # list of tuple (a, b, type) where the canonical order is used
        self.src = src
        self.dst = dst
        self.dummy = False
        if len(self.edges) == 0:
            self.dummy = True
    
    def get_edges(self):
        return copy(self.edges)
    
    def get_num_edges(self):
        return len(self.edges)

    def __str__(self):
        return f'''
                    src subgraph: {self.src}, dst subgraph: {self.dst},
                    atom bonds: {self.edges}
                '''


class Molecule(nx.Graph):
    '''molecule represented in subgraph-level'''

    def __init__(self, mol: Union[str, RDKitMol]=None, groups: list=None, kekulize: bool=False):
        super().__init__()
        if mol is None:
            return

        if isinstance(mol, str):
            smiles, rdkit_mol = mol, smi2mol(mol, kekulize)
        else:
            smiles, rdkit_mol = mol2smi(mol), mol
        self.graph['smiles'] = smiles
        # processing atoms
        aid2pos = {}
        for pos, group in enumerate(groups):
            for aid in group:
                aid2pos[aid] = pos
            subgraph_mol = get_submol(rdkit_mol, group, kekulize)
            if subgraph_mol is None:
                print(mol.GetAtomWithIdx(group[0]).GetSymbol())
                print('!!!!!!!!!!!!!!!!!!!!',rdkit_mol, group, kekulize)
            subgraph_smi = mol2smi(subgraph_mol)
            atom_mapping = get_submol_atom_map(rdkit_mol, subgraph_mol, group, kekulize)
            node = SubgraphNode(subgraph_smi, pos, atom_mapping, kekulize)
            self.add_node(node)
        # process edges
        edges_arr = [[[] for _ in groups] for _ in groups]  # adjacent
        for edge_idx in range(rdkit_mol.GetNumBonds()):
            bond = rdkit_mol.GetBondWithIdx(edge_idx)
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            begin_subgraph_pos = aid2pos[begin]
            end_subgraph_pos = aid2pos[end]
            begin_mapped = self.nodes[begin_subgraph_pos]['subgraph'].atom_mapping[begin]
            end_mapped = self.nodes[end_subgraph_pos]['subgraph'].atom_mapping[end]

            bond_type = bond.GetBondType()
            edges_arr[begin_subgraph_pos][end_subgraph_pos].append((begin_mapped, end_mapped, bond_type))
            edges_arr[end_subgraph_pos][begin_subgraph_pos].append((end_mapped, begin_mapped, bond_type))

        # add egdes into the graph
        for i in range(len(groups)):
            for j in range(len(groups)):
                if not i < j or len(edges_arr[i][j]) == 0:
                    continue
                edge = SubgraphEdge(i, j, edges_arr[i][j])
                self.add_edge(edge)
    
    @classmethod
    def from_nx_graph(cls, graph: nx.Graph, deepcopy=True):
        if deepcopy:
            graph = deepcopy(graph)
        graph.__class__ = Molecule
        return graph

    @classmethod
    def merge(cls, mol0, mol1, edge=None):
        # reorder
        node_mappings = [{}, {}]
        mols = [mol0, mol1]
        mol = Molecule.from_nx_graph(nx.Graph())
        for i in range(2):
            for n in mols[i].nodes:
                node_mappings[i][n] = len(node_mappings[i])
                node = deepcopy(mols[i].get_node(n))
                node.pos = node_mappings[i][n]
                mol.add_node(node)
            for src, dst in mols[i].edges:
                edge = deepcopy(mols[i].get_edge(src, dst))
                edge.src = node_mappings[i][src]
                edge.dst = node_mappings[i][dst]
                mol.add_edge(src, dst, connects=edge)
        # add new edge
        edge = deepcopy(edge)
        edge.src = node_mappings[0][edge.src]
        edge.dst = node_mappings[1][edge.dst]
        mol.add_edge(edge)
        return mol

    def get_edge(self, i, j) -> SubgraphEdge:
        return self[i][j]['connects']
    
    def get_node(self, i) -> SubgraphNode:
        return self.nodes[i]['subgraph']

    def add_edge(self, edge: SubgraphEdge) -> None:
        src, dst = edge.src, edge.dst
        super().add_edge(src, dst, connects=edge)
    
    def add_node(self, node: SubgraphNode) -> None:
        n = node.pos
        super().add_node(n, subgraph=node)

    def subgraph(self, nodes: list):
        graph = super().subgraph(nodes)
        assert isinstance(graph, Molecule)
        return graph

    def to_rdkit_mol(self):
        mol = Chem.RWMol()
        aid_mapping, order = {}, []
        # add all the subgraphs to rwmol
        for n in self.nodes:
            subgraph = self.get_node(n)
            submol = subgraph.get_mol()
            local2global = {}
            for global_aid in subgraph.atom_mapping:
                local_aid = subgraph.atom_mapping[global_aid]
                local2global[local_aid] = global_aid
            for atom in submol.GetAtoms():
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                mol.AddAtom(atom)
                aid_mapping[(n, atom.GetIdx())] = len(aid_mapping)
                order.append(local2global[atom.GetIdx()])
            for bond in submol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                begin, end = aid_mapping[(n, begin)], aid_mapping[(n, end)]
                mol.AddBond(begin, end, bond.GetBondType())
        for src, dst in self.edges:
            subgraph_edge = self.get_edge(src, dst)
            pid_src, pid_dst = subgraph_edge.src, subgraph_edge.dst
            for begin, end, bond_type in subgraph_edge.edges:
                begin, end = aid_mapping[(pid_src, begin)], aid_mapping[(pid_dst, end)]
                mol.AddBond(begin, end, bond_type)
        mol = mol.GetMol()
        new_order = [-1 for _ in order]
        for cur_i, ordered_i in enumerate(order):
            new_order[ordered_i] = cur_i
        mol = Chem.RenumberAtoms(mol, new_order)
        # sanitize, we need to handle mal-formed N+
        mol.UpdatePropertyCache(strict=False)
        ps = Chem.DetectChemistryProblems(mol)
        if not ps:  # no problem
            Chem.SanitizeMol(mol)
            return mol
        for p in ps:
            if p.GetType()=='AtomValenceException':  # for N+, we need to set its formal charge
                at = mol.GetAtomWithIdx(p.GetAtomIdx())
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                    at.SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        return mol

    def to_SVG(self, path: str, size: tuple=(200, 200), add_idx=False) -> str:
        # save the subgraph-level molecule to an SVG image
        # return the content of svg in string format
        mol = self.to_rdkit_mol()
        if add_idx:  # this will produce an ugly figure
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                atom.SetAtomMapNum(i)
        tm = rdMolDraw2D.PrepareMolForDrawing(mol)
        view = rdMolDraw2D.MolDraw2DSVG(*size)
        option = view.drawOptions()
        option.legendFontSize = 18
        option.bondLineWidth = 1
        option.highlightBondWidthMultiplier = 20
        sg_atoms, sg_bonds = [], []
        atom2subgraph, atom_color, bond_color = {}, {}, {}
        # atoms in each subgraph
        for i in self.nodes:
            node = self.get_node(i)
            # random color in rgb. mix with white to obtain soft colors
            color = tuple(((np.random.rand(3) + 1)/ 2).tolist())
            for atom_id in node.atom_mapping:
                sg_atoms.append(atom_id)
                atom2subgraph[atom_id] = i
                atom_color[atom_id] = color
        # bonds in each subgraph
        for bond_id in range(mol.GetNumBonds()):
            bond = mol.GetBondWithIdx(bond_id)
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if atom2subgraph[begin] == atom2subgraph[end]:
                sg_bonds.append(bond_id)
                bond_color[bond_id] = atom_color[begin]
        view.DrawMolecules([tm], highlightAtoms=[sg_atoms], \
                           highlightBonds=[sg_bonds], highlightAtomColors=[atom_color], \
                           highlightBondColors=[bond_color])
        view.FinishDrawing()
        svg = view.GetDrawingText()
        with open(path, 'w') as fout:
            fout.write(svg)
        return svg

    def to_smiles(self):
        rdkit_mol = self.to_rdkit_mol()
        return mol2smi(rdkit_mol)

    def __str__(self):
        desc = 'nodes: \n'
        for ni, node in enumerate(self.nodes):
            desc += f'{ni}:{self.get_node(node)}\n'
        desc += 'edges: \n'
        for src, dst in self.edges:
            desc += f'{src}-{dst}:{self.get_edge(src, dst)}\n'
        return desc


MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':5, 'O':2, 'P':5, 'S':6} #, 'Si':4}
def smi2mol(smiles: str, kekulize=False, sanitize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        mol = Chem.MolFromSmiles(f'[{smiles}]', sanitize=False)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol, canonical=True):
    return Chem.MolToSmiles(mol, canonical=canonical)


def get_submol(mol, atom_indices, kekulize=False):
    if len(atom_indices) == 1:
        atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
       
        return smi2mol(atom_symbol, kekulize)
    aid_dict = { i: True for i in atom_indices }
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid = bond.GetBeginAtomIdx()
        end_aid = bond.GetEndAtomIdx()
        if begin_aid in aid_dict and end_aid in aid_dict:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol
'''classes below are used for principal subgraph extraction'''

def get_submol_atom_map(mol, submol, group, kekulize=False):
    if len(group) == 1:
        return { group[0]: 0 }
    # turn to smiles order
    smi = mol2smi(submol)
    submol = smi2mol(smi, kekulize, sanitize=False)
    # # special with N+ and N-
    # for atom in submol.GetAtoms():
    #     if atom.GetSymbol() != 'N':
    #         continue
    #     if (atom.GetExplicitValence() == 3 and atom.GetFormalCharge() == 1) or atom.GetExplicitValence() < 3:
    #         atom.SetNumRadicalElectrons(0)
    #         atom.SetNumExplicitHs(2)
    
    matches = mol.GetSubstructMatches(submol)
    old2new = { i: 0 for i in group }  # old atom idx to new atom idx
    found = False
    for m in matches:
        hit = True
        for i, atom_idx in enumerate(m):
            if atom_idx not in old2new:
                hit = False
                break
            old2new[atom_idx] = i
        if hit:
            found = True
            break
    assert found
    return old2new

def cnt_atom(smi, return_dict=False):
    atom_dict = { atom: 0 for atom in MAX_VALENCE }
    for i in range(len(smi)):
        symbol = smi[i].upper()
        next_char = smi[i+1] if i+1 < len(smi) else None
        if symbol == 'B' and next_char == 'r':
            symbol += next_char
        elif symbol == 'C' and next_char == 'l':
            symbol += next_char
        if symbol in atom_dict:
            atom_dict[symbol] += 1
    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())


class MolInSubgraph:
    def __init__(self, mol, kekulize=False):
        self.mol = mol
        self.smi = mol2smi(mol)
        self.kekulize = kekulize
        self.subgraphs, self.subgraphs_smis = {}, {}  # pid is the key (init by all atom idx)
        for atom in mol.GetAtoms():
            idx, symbol = atom.GetIdx(), atom.GetSymbol()
            self.subgraphs[idx] = { idx: symbol }
            self.subgraphs_smis[idx] = symbol
        self.inversed_index = {} # assign atom idx to pid
        self.upid_cnt = len(self.subgraphs)
        for aid in range(mol.GetNumAtoms()):
            for key in self.subgraphs:
                subgraph = self.subgraphs[key]
                if aid in subgraph:
                    self.inversed_index[aid] = key
        self.dirty = True
        self.smi2pids = {} # private variable, record neighboring graphs and their pids

    def get_nei_subgraphs(self):
        nei_subgraphs, merge_pids = [], []
        for key in self.subgraphs:
            subgraph = self.subgraphs[key]
            local_nei_pid = []
            for aid in subgraph:
                atom = self.mol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if nei_idx in subgraph or nei_idx > aid:   # only consider connecting to former atoms
                        continue
                    local_nei_pid.append(self.inversed_index[nei_idx])
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                new_subgraph = copy(subgraph)
                new_subgraph.update(self.subgraphs[nei_pid])
                nei_subgraphs.append(new_subgraph)
                merge_pids.append((key, nei_pid))
        return nei_subgraphs, merge_pids
    
    def get_nei_smis(self):
        if self.dirty:
            nei_subgraphs, merge_pids = self.get_nei_subgraphs()
            nei_smis, self.smi2pids = [], {}
            for i, subgraph in enumerate(nei_subgraphs):
                submol = get_submol(self.mol, list(subgraph.keys()), kekulize=self.kekulize)
                smi = mol2smi(submol)
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, [])
                self.smi2pids[smi].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis

    def merge(self, smi):
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            merge_pids = self.smi2pids[smi]
            for pid1, pid2 in merge_pids:
                if pid1 in self.subgraphs and pid2 in self.subgraphs: # possibly del by former
                    self.subgraphs[pid1].update(self.subgraphs[pid2])
                    self.subgraphs[self.upid_cnt] = self.subgraphs[pid1]
                    self.subgraphs_smis[self.upid_cnt] = smi
                    # self.subgraphs_smis[pid1] = smi
                    for aid in self.subgraphs[pid2]:
                        self.inversed_index[aid] = pid1
                    for aid in self.subgraphs[pid1]:
                        self.inversed_index[aid] = self.upid_cnt
                    del self.subgraphs[pid1]
                    del self.subgraphs[pid2]
                    del self.subgraphs_smis[pid1]
                    del self.subgraphs_smis[pid2]
                    self.upid_cnt += 1
        self.dirty = True   # mark the graph as revised

    def get_smis_subgraphs(self):
        # return list of tuple(smi, idxs)
        res = []
        for pid in self.subgraphs_smis:
            smi = self.subgraphs_smis[pid]
            group_dict = self.subgraphs[pid]
            idxs = list(group_dict.keys())
            res.append((smi, idxs))
        return res


def freq_cnt(mol):
    freqs = {}
    nei_smis = mol.get_nei_smis()
    for smi in nei_smis:
        freqs.setdefault(smi, 0)
        freqs[smi] += 1
    return freqs, mol

def graph_bpe_smiles(smis, vocab_len, vocab_path, cpus, kekulize):
    # init to atoms
    mols = []
    for smi in tqdm(smis):
        try:
            mol = MolInSubgraph(smi2mol(smi, kekulize), kekulize)
            mols.append(mol)
        except Exception as e:
            print(f'Parsing {smi} failed. Skip.', level='ERROR')
    # loop
    selected_smis, details = list(MAX_VALENCE.keys()), {}   # details: <smi: [atom cnt, frequency]
    # calculate single atom frequency
    for atom in selected_smis:
        details[atom] = [1, 0]  # frequency of single atom is not calculated
    for smi in smis:
        cnts = cnt_atom(smi, return_dict=True)
        for atom in details:
            if atom in cnts:
                details[atom][1] += cnts[atom]
    # bpe process
    add_len = vocab_len - len(selected_smis)
    print(f'Added {len(selected_smis)} atoms, {add_len} principal subgraphs to extract')
    pbar = tqdm(total=add_len)
    pool = mp.Pool(cpus)
    while len(selected_smis) < vocab_len:
        res_list = pool.map(freq_cnt, mols)  # each element is (freq, mol) (because mol will not be synced...)
        freqs, mols = {}, []
        for freq, mol in res_list:
            mols.append(mol)
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]
        # find the subgraph to merge
        max_cnt, merge_smi = 0, ''
        for smi in freqs:
            cnt = freqs[smi]
            if cnt > max_cnt:
                max_cnt = cnt
                merge_smi = smi
        # merge
        for mol in mols:
            mol.merge(merge_smi)
        if merge_smi in details:  # corner case: re-extracted from another path
            continue
        selected_smis.append(merge_smi)
        details[merge_smi] = [cnt_atom(merge_smi), max_cnt]
        pbar.update(1)
    pbar.close()
    print('sorting vocab by atom num')
    selected_smis.sort(key=lambda x: details[x][0], reverse=True)
    pool.close()
    with open(vocab_path, 'w') as fout:
        fout.write(json.dumps({'kekulize': kekulize}) + '\n')
        fout.writelines(list(map(lambda smi: f'{smi}\t{details[smi][0]}\t{details[smi][1]}\n', selected_smis)))

def graph_bpe(fname, vocab_len, vocab_path, cpus, kekulize):
    # load molecules
    print(f'Loading mols from {fname} ...')
    with open(fname, 'r') as fin:
        smis = list(map(lambda x: x.strip(), fin.readlines()))
    # init to atoms
    mols = []
    for smi in tqdm(smis):
        try:
            mol = MolInSubgraph(smi2mol(smi, kekulize), kekulize)
            mols.append(mol)
        except Exception as e:
            print(f'Parsing {smi} failed. Skip.', level='ERROR')
    # loop
    selected_smis, details = list(MAX_VALENCE.keys()), {}   # details: <smi: [atom cnt, frequency]
    # calculate single atom frequency
    for atom in selected_smis:
        details[atom] = [1, 0]  # frequency of single atom is not calculated
    for smi in smis:
        cnts = cnt_atom(smi, return_dict=True)
        for atom in details:
            if atom in cnts:
                details[atom][1] += cnts[atom]
    # bpe process
    add_len = vocab_len - len(selected_smis)
    print(f'Added {len(selected_smis)} atoms, {add_len} principal subgraphs to extract')
    pbar = tqdm(total=add_len)
    pool = mp.Pool(cpus)
    while len(selected_smis) < vocab_len:
        res_list = pool.map(freq_cnt, mols)  # each element is (freq, mol) (because mol will not be synced...)
        freqs, mols = {}, []
        for freq, mol in res_list:
            mols.append(mol)
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]
        # find the subgraph to merge
        max_cnt, merge_smi = 0, ''
        for smi in freqs:
            cnt = freqs[smi]
            if cnt > max_cnt:
                max_cnt = cnt
                merge_smi = smi
        # merge
        for mol in mols:
            mol.merge(merge_smi)
        if merge_smi in details:  # corner case: re-extracted from another path
            continue
        selected_smis.append(merge_smi)
        details[merge_smi] = [cnt_atom(merge_smi), max_cnt]
        pbar.update(1)
    pbar.close()
    print('sorting vocab by atom num')
    selected_smis.sort(key=lambda x: details[x][0], reverse=True)
    pool.close()
    with open(vocab_path, 'w') as fout:
        fout.write(json.dumps({'kekulize': kekulize}) + '\n')
        fout.writelines(list(map(lambda smi: f'{smi}\t{details[smi][0]}\t{details[smi][1]}\n', selected_smis)))
    return selected_smis, details


class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        # load kekulize config
        config = json.loads(lines[0])
        self.kekulize = config['kekulize']
        lines = lines[1:]
        
        self.vocab_dict = {}
        self.idx2subgraph, self.subgraph2idx = [], {}
        self.max_num_nodes = 0
        for line in lines:
            smi, atom_num, freq = line.strip().split('\t')
            self.vocab_dict[smi] = (int(atom_num), int(freq))
            self.max_num_nodes = max(self.max_num_nodes, int(atom_num))
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)
        print(len(self.subgraph2idx))
        self.pad, self.end = '<pad>', '<s>'
        for smi in [self.pad, self.end]:
            self.subgraph2idx[smi] = len(self.idx2subgraph)
            self.idx2subgraph.append(smi)
        # for fine-grained level (atom level)
        self.bond_start = '<bstart>'
        self.max_num_nodes += 2 # start, padding
    
    def tokenize(self, mol):
        # smiles = mol
        # if isinstance(mol, str):
        #     mol = smi2mol(mol, self.kekulize)
        # else:
        #     smiles = mol2smi(mol)
        rdkit_mol = mol
        mol = MolInSubgraph(mol, kekulize=self.kekulize)
        while True:
            nei_smis = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''
            for smi in nei_smis:
                if smi not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi
            if max_freq == -1:
                break
            mol.merge(merge_smi)
        res = mol.get_smis_subgraphs()
        # construct reversed index
        aid2pid = {}
        for pid, subgraph in enumerate(res):
            _, aids = subgraph
            for aid in aids:
                aid2pid[aid] = pid
        # construct adjacent matrix
        ad_mat = [[0 for _ in res] for _ in res]
        for aid in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(aid)
            for nei in atom.GetNeighbors():
                nei_id = nei.GetIdx()
                i, j = aid2pid[aid], aid2pid[nei_id]
                if i != j:
                    ad_mat[i][j] = ad_mat[j][i] = 1
        group_idxs = [x[1] for x in res]
        return Molecule(rdkit_mol, group_idxs, self.kekulize)

    def idx_to_subgraph(self, idx):
        return self.idx2subgraph[idx]
    
    def subgraph_to_idx(self, subgraph):
        return self.subgraph2idx[subgraph]
    
    def pad_idx(self):
        return self.subgraph2idx[self.pad]
    
    def end_idx(self):
        return self.subgraph2idx[self.end]
    
    def atom_vocab(self):
        return copy(self.atom_level_vocab)

    def num_subgraph_type(self):
        return len(self.idx2subgraph)
    
    def atom_pos_pad_idx(self):
        return self.max_num_nodes - 1
    
    def atom_pos_start_idx(self):
        return self.max_num_nodes - 2

    def __call__(self, mol):
        return self.tokenize(mol)
    
    def __len__(self):
        return len(self.idx2subgraph)

def parse():
    parser = argparse.ArgumentParser(description='Principal subgraph extraction motivated by bpe')
    parser.add_argument('--smiles', type=str, default='COc1cc(C=NNC(=O)c2ccc(O)cc2O)ccc1OCc1ccc(Cl)cc1',
                        help='The molecule to tokenize (example)')
    parser.add_argument('--data', type=str, required=True, help='Path to molecule corpus')
    parser.add_argument('--vocab_size', type=int, default=500, help='Length of vocab')
    parser.add_argument('--output', type=str, required=True, help='Path to save vocab')
    parser.add_argument('--workers', type=int, default=16, help='Number of cpus to use')
    parser.add_argument('--kekulize', action='store_true', help='Whether to kekulize the molecules (i.e. replace aromatic bonds with alternating single and double bonds)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    graph_bpe(args.data, vocab_len=args.vocab_size, vocab_path=args.output,
              cpus=args.workers, kekulize=args.kekulize)
    tokenizer = Tokenizer(args.output)
    print(f'Example: {args.smiles}')
    mol = tokenizer.tokenize(args.smiles)
    print('Tokenized mol: ')
    print(mol)
    print('Reconstruct smiles to make sure it is right: ')
    smi = mol.to_smiles()
    print(smi)
    assert smi == args.smiles
    print('Assertion test passed')
    mol.to_SVG('example.svg')
