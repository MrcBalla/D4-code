from typing import Dict, List, Tuple, Optional, Union
import re

import copy
from rdkit import Chem

import torch
from torch import Tensor
from src.datatypes.sparse import SparseGraph

from src.datatypes.sparse import to_directed
from rdkit.Geometry import Point3D
import logging

from rdkit.Chem.rdchem import BondType as BT
import networkx as nx
import numpy as np
from src.data.simple_transforms import batched
from sklearn.manifold import MDS


from rdkit import DistanceGeometry

BOND_TYPES_REAL = {1: BT.SINGLE, 2: BT.DOUBLE, 3: BT.TRIPLE, 11: BT.AROMATIC}
BOND_TYPES = {bt: str(bt) for bt in [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]}
BOND_TYPES_REV = {str(bt): bt for bt in [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]}
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

@batched
def mol2smiles(mol, sanitize=True, canonical=True, isomeric_smile=False):
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smile, canonical = canonical)

    return smiles

@batched
def smiles2mol(smiles):
    return Chem.MolFromSmiles(smiles)

def pos_from_dist_MDF(dist_matrix):
    dist_matrix = dist_matrix.numpy()
    mds = MDS(n_components=3, max_iter=1000)
    pos=mds.fit_transform(dist_matrix)

    return pos

def pos_from_dist_rdkit(dist_matrix):
    return DistanceGeometry.DistGeom.EmbedBoundsMatrix(dist_matrix.double().numpy())

def pos_from_dist(dist_matrix):
    M_matrix=[]

    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[0]):
            M_matrix.append(0.5*(dist_matrix[0,j]**2 + dist_matrix[i,0]**2 - dist_matrix[i,j]**2).item())
        
    M_matrix = torch.tensor(M_matrix).reshape(dist_matrix.shape[0], dist_matrix.shape[0])
    
    eig_val, eig_vec = torch.linalg.eig(M_matrix)
    
    eig_value_chosen=[]
    index_elem=[]
    resorted = torch.sort(eig_val.float(), descending=True)
    for elem in range(3):
        if resorted[0][elem]>0:
            eig_value_chosen.append(resorted[0][elem].item())
            index_elem.append(resorted[1][elem].item())
    if eig_value_chosen == []:
        return None
    
    eig_vector_chosen=eig_vec[:,index_elem].float()
    eig_value_chosen=torch.tensor(eig_value_chosen, device=eig_vector_chosen.device).float()
    pos = np.zeros((dist_matrix.shape[0],3))
    
    for i in [0,1,2]:
        pos[:,i]=(eig_value_chosen[i].float()*eig_vector_chosen[:,i].float()).numpy()
    
    return pos



@batched
def mol2nx(mol):

    G = nx.Graph()

    if mol!=None:
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                        label=atom.GetSymbol())
            #    atomic_num=atom.GetAtomicNum(),
            #    formal_charge=atom.GetFormalCharge(),
            #    chiral_tag=atom.GetChiralTag(),
            #    hybridization=atom.GetHybridization(),
            #    num_explicit_hs=atom.GetNumExplicitHs(),
            #    is_aromatic=atom.GetIsAromatic())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        label=int(bond.GetBondTypeAsDouble()))
            #    bond_type=bond.GetBondType())

        return G
    else:
        pass

from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation as R

def check_position(mol_before, mol_after):
    conf = mol_before.GetConformer()
    positions_before = conf.GetPositions()
        
    conf=mol_after.GetConformer()
    positions_after = conf.GetPositions()
    
    return R.align_vectors(positions_before, positions_after)[1]
    
        

def build_molecule(
        atom_types: Tensor,
        charges: Tensor,
        pos: Tensor, 
        edge_index: Tensor,
        edge_types: Tensor,
        atom_decoder: Union[Dict[int, str], Dict[str, int]],
        bond_decoder: Optional[Dict[int, str]]=None,
        relaxed: bool=False,
        verbose: bool=False,
        distance: Tensor=None
    ) -> Chem.Mol:
    if verbose:
        print("building new molecule")

    atom_decoder, bond_decoder = check_atom_bond_decoders(
        atom_decoder,
        bond_decoder
    )

    ###############################  PARSE ATOMS  ##############################
    mol = Chem.RWMol()
    for (atom, charge) in zip(atom_types, charges):
        a = Chem.Atom(atom_decoder[atom.item()])
        if charge.item() != 0:
                a.SetFormalCharge(int(charge.item()))
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    ###############################  PARSE BONDS  ##############################

    for bond, link in zip(edge_types, edge_index.permute(1, 0)):

        if link[0].item() != link[1].item():
            mol.AddBond(link[0].item(), link[1].item(), BOND_TYPES_REV[bond_decoder[bond.item()]])
            if verbose:
                print(
                    "bond added:", link[0].item(), link[1].item(), bond.item(),
                      bond_decoder[bond.item()]
                )
    
    try:
            mol = mol.GetMol()
    except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None
    
    if pos != None:
        positions = pos.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        mol.AddConformer(conf)
    
    if distance != None:
        try:
            pos = pos_from_dist_rdkit(distance.reshape(len(atom_types), len(atom_types)))
            pos= pos-np.mean(pos,axis=0)

            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                            conf.SetAtomPosition(i, Point3D(pos[i][0], pos[i][1], pos[i][2]))
            mol.AddConformer(conf)
        except Exception as e:
            pass
    return mol



def build_graph_from_molecule(
        mol: Chem.Mol,
        atom_encoder: Union[Dict[int, str], Dict[str, int]],
        bond_encoder: Union[Dict[int, str], Dict[str, int]]
    ):

    atom_encoder, bond_encoder = check_atom_bond_decoders(
        atom_encoder,
        bond_encoder,
        atom_idx_to_name=False
    )

    # build graph from molecule
    atom_types = []
    edge_index = []
    edge_types = []

    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])


    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = str(bond.GetBondType())

        edge_index.append([start, end])
        edge_types.append(bond_encoder[bond_type])

    # tranform to tensor
    x = torch.tensor(atom_types, dtype=torch.long)
    if len(edge_index) == 0: # if no edges, then set special case
        edge_index = torch.tensor([[], []], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).permute(1, 0)
    edge_attr = torch.tensor(edge_types, dtype=torch.long)

    g = SparseGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return g


# from GDSS
def valid_mol_can_with_seg(x, largest_connected_comp=True):
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


class GraphToMoleculeConverter:

    def __init__(
            self,
            atom_decoder: Union[Dict[int, str], Dict[str, int]],
            bond_decoder: Union[Dict[int, str], Dict[str, int]],
            relaxed: bool=False,
            post_hoc_mols_fix: bool=False,
            post_hoc_mols_convert: bool=False
        ):

        self.atom_decoder, self.bond_decoder = check_atom_bond_decoders(
            atom_decoder,
            bond_decoder
        )
        self.relaxed = relaxed
        self.post_hoc_mols_fix = post_hoc_mols_fix
        self.post_hoc_mols_convert = post_hoc_mols_convert


    def __call__(
            self,
            batch: Union[List[SparseGraph], SparseGraph],
            override_relaxed: Optional[bool]=None,
            override_post_hoc_mols_fix: Optional[bool]=None,
            override_post_hoc_mols_convert: Optional[bool]=None
        ) -> Union[List[Chem.Mol], Chem.Mol]:

        is_batch = True

        if not isinstance(batch, list):
            # check and format for batch
            is_batch = hasattr(batch, 'ptr')

            if is_batch:
                batch = batch.to_data_list()
            else:
                batch = [batch]

        # build molecules from graphs
        out_molecules = []
        rss=[]
        g: SparseGraph
        d=[]
        for g in batch:

            g = g.clone()

            if g.is_undirected():
                g.edge_index, g.edge_attr = to_directed(g.edge_index, g.edge_attr)

            # collapse classes if needed
            g.collapse()

            mol = build_molecule(
                atom_types =		g.x,
                pos =               g.pos,
                charges=            g.attribute_node,
                edge_index =		g.edge_index,
                edge_types =		g.edge_attr,
                atom_decoder =		self.atom_decoder,
                bond_decoder =		self.bond_decoder,
                relaxed =		    self.relaxed if override_relaxed is None else override_relaxed
            )

            # apply correction if needed
            #mol = self.correct_mol(mol, override_post_hoc_mols_fix)

            # apply conversion if needed
            #mol = self.convert_mol_and_back(mol, override_post_hoc_mols_convert)

            out_molecules.append(mol)
            
        return out_molecules if is_batch else out_molecules[0]
    
    def graph_to_molecule(
            self,
            batch: Union[List[SparseGraph], SparseGraph],
            override_relaxed: Optional[bool]=None,
            override_post_hoc_mols_fix: Optional[bool]=None,
            override_post_hoc_mols_convert: Optional[bool]=None
        ) -> Union[List[Chem.Mol], Chem.Mol]:

        return self(
            batch,
            override_relaxed,
            override_post_hoc_mols_fix,
            override_post_hoc_mols_convert
        )
            

    def molecule_to_graph(
            self,
            mols: Union[List[Chem.Mol], List[str], Chem.Mol, str],
            kekulize: bool=True,
            override_post_hoc_mols_fix: Optional[bool]=None,
            override_post_hoc_mols_convert: Optional[bool]=None
        ) -> SparseGraph:

        single_mol = False

        # check element
        if isinstance(mols, (str, Chem.Mol)):
            mols = [mols]
            single_mol = True

        out_graphs = []

        # start conversion
        for mol in mols:

            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)

            if kekulize:
                mol = copy.deepcopy(mol)
                Chem.Kekulize(mol)

            # apply correction if needed
            mol = self.correct_mol(mol, override_post_hoc_mols_fix)

            # apply conversion if needed
            mol = self.convert_mol_and_back(mol, override_post_hoc_mols_convert)

            # build graph from molecule
            g = build_graph_from_molecule(
                mol,
                atom_encoder = self.atom_decoder,
                bond_encoder = self.bond_decoder,
            )

            out_graphs.append(g)

        return out_graphs[0] if single_mol else out_graphs
    

    
    def correct_mol(
            self,
            mol: Chem.Mol,
            override_post_hoc_mols_fix: Optional[bool]=None
        ):

        if override_post_hoc_mols_fix is None:
            if self.post_hoc_mols_fix:
                mol, _ = correct_mol(mol)

        else:
            if override_post_hoc_mols_fix:
                mol, _ = correct_mol(mol)

        return mol
    
    def convert_mol_and_back(
            self,
            mol: Chem.Mol,
            override_post_hoc_mols_convert: Optional[bool]=None
        ):

        if override_post_hoc_mols_convert is None:
            if self.post_hoc_mols_convert:
                mol = valid_mol_can_with_seg(mol)

        else:
            if override_post_hoc_mols_convert:
                mol = valid_mol_can_with_seg(mol)

        return mol



def check_atom_bond_decoders(
        atom_decoder,
        bond_decoder,
        atom_idx_to_name: bool=True,
    ):

    if atom_idx_to_name:
        type_key = int
    else:
        type_key = str

    # if atom_idx_to_name is True:
    # if atom_decoder is given as atom_name -> atom_idx
    # reverse mapping to atom_idx -> atom_name
    # else: do the same with atom_idx -> atom_name
    if len(atom_decoder) > 0 and not isinstance(next(iter(atom_decoder.keys())), type_key):
        atom_decoder = {v: k for k, v in atom_decoder.items()}
    
    if len(bond_decoder) > 0 and not isinstance(next(iter(bond_decoder.keys())), type_key):
        bond_decoder = {v: k for k, v in bond_decoder.items()}

    return atom_decoder, bond_decoder


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence
    


# code from: https://github.com/GRAPH-0/CDGS/blob/main/utils.py
def correct_mol(
        mol: Chem.Mol,
    ):

    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            queue = []

            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, BOND_TYPES_REAL[t])

    return mol, no_correct