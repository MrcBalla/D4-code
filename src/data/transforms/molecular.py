from typing import Union, List, Dict, Callable, Optional

from tqdm import tqdm

import locale

from . import DFBaseTransform
from .conditions import Condition
from .preprocessing import (
    CSV_TABLE
)

import math
import os

import torch
from torch_geometric.utils import subgraph

from tqdm import tqdm

import rdkit
from rdkit.Chem import rdMolDescriptors
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

from src.datatypes.sparse import SparseGraph
from src.data.simple_transforms.molecular import (
    GraphToMoleculeConverter,
    mol2smiles,
    mol2nx
)
from src.data.simple_transforms.chemical_props import (
    PROPS_NAME_TO_FUNC
)

global dictionary_edge_features
dictionary_edge_features={}



class MolecularPipelineException(Exception):
    pass


class DFApplyUnitConversion(DFBaseTransform):

    def __init__(
            self,
            datafield: str,
            conversion_factors: torch.Tensor
        ):

        self.datafield = datafield
        self.conversion_factors = conversion_factors

    
    def __call__(self, data: Dict) -> Dict:

        # get table of csv datafield
        table = data[self.datafield]
        table = torch.cat([table[:, 3:], table[:, :3]], dim=-1)

        # apply conversions to each row
        table = table * self.conversion_factors.view(1, -1)

        data[self.datafield] = table

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datafield]
    @property
    def output_df_list(self) -> List[str]:
        return [self.datafield]
    
    def args_repr(self) -> str:
        return (
            f'factors shape={self.conversion_factors.shape}'
        )

class DFAddFeaturesSDF(DFBaseTransform):
    def __init__(
            self,
            datafield: str,
            remove_hs: bool
    ):
        self.file_data=datafield
        self.remove_hs=remove_hs

    def __call__(self,data):

        filename: str = data[self.file_data]
        print('Starting Calculating Chemical Features ...')

        if not self.remove_hs:
            if os.path.isfile('datasets/qm9/raw/gdb9_feat_hs.sdf'):
                print('Dataset alreay transformed.')
                data['mols_file']='datasets/qm9/raw/gdb9_feat_hs.sdf'
                return data
        else:
             if os.path.isfile('datasets/qm9/raw/gdb9_feat.sdf'):
                print('Dataset alreay transformed.')
                data['mols_file']='datasets/qm9/raw/gdb9_feat.sdf'
                return data
        if not self.remove_hs:
            suppl=Chem.SDMolSupplier(filename, sanitize=False, removeHs=False)
            w=Chem.SDWriter(data['root_path']+"/raw/gdb9_feat_hs.sdf")
            mols = [mol for mol in suppl if mol is not None]
        else:
            suppl=Chem.SDMolSupplier(filename, sanitize=False)
            w=Chem.SDWriter(data['root_path']+"/raw/gdb9_feat.sdf")
            mols = [mol for mol in suppl if mol is not None]
            mols = [Chem.RemoveHs(mol, sanitize=False) for mol in suppl if mol is not None]

        # compute the distances for each couple of atoms
        for mol in tqdm(mols):
            # compute distance between atoms 
            mol.GetConformer()
            distance=[]
            #log_p_atom=[]
            #MR_list=[]
            #atomic_number=[]
            formal_charge=[]
            coordinates=[]

            for i,atom in enumerate(mol.GetAtoms()):

                # single first atom position
                pos_i = mol.GetConformer().GetAtomPosition(i)
                formal_charge.append(atom.GetFormalCharge())

                # coordinate of first atom
                point_i=np.array([pos_i.x, pos_i.y, pos_i.z])
                if len(coordinates)==0:
                    coordinates=point_i
                else:
                    coordinates = np.concatenate((coordinates, point_i))
                for j,atom in enumerate(mol.GetAtoms()):
                    pos_j = mol.GetConformer().GetAtomPosition(j)
                    point_j=np.array([pos_j.x, pos_j.y, pos_j.z])
                    distance.append(np.linalg.norm(point_i-point_j))


            # convert all the quantity to string, necessary for writing in sdf file
            distance_str = ' '.join(map(str, distance))
            formal_charge_str = ' '.join(map(str, formal_charge))
            coordinates_str = ' '.join(map(str, coordinates))

            mol.SetProp("Distance", distance_str)
            mol.SetProp("Formal_Charge", formal_charge_str)
            mol.SetProp('Coordinates', coordinates_str)
        
        
        # save modifications
        for mol in mols:
            w.write(mol)
        w.close()

        print('End Calculations')

        # change data values
        if not self.remove_hs:
            data['mols_file']='datasets/qm9/raw/gdb9_feat_hs.sdf'
        else:
            data['mols_file']='datasets/qm9/raw/gdb9_feat.sdf'

        return data


class DFReadMolecules(DFBaseTransform):

    def __init__(
            self,
            datafield: str,
            new_datafield: str,
            sanitize: bool=False,
            remove_hydrogens: bool=False
        ):

        self.datafield = datafield
        self.new_datafield = new_datafield
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens

    
    def __call__(self, data: Dict) -> Dict:

        filename: str = data[self.datafield]

        ############  molecules file  ############
        if filename.endswith('.sdf'):

            suppl = Chem.SDMolSupplier(
                filename,
                removeHs=self.remove_hydrogens,
                sanitize=self.sanitize
            )

        #############  smiles file  ##############
        elif filename.endswith('.smi'):
            
            suppl = Chem.SmilesMolSupplier(
                filename
            )

        else:
            raise NotImplementedError(
                f'Molecules supplier for file {filename} not implemented'
            )
        
        data[self.new_datafield] = suppl

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datafield]
    @property
    def output_df_list(self) -> List[str]:
        return [self.new_datafield]
    
    def args_repr(self) -> str:
        return (
            f'sanitize={self.sanitize}\n'
            f'remove_hydrogens={self.remove_hydrogens}'
        )
    

class DFAtomsPositions(DFBaseTransform):
    pass


class DFMoleculePreprocess(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            kekulize: bool=False,
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df
        # apply kekulization
        self.kekulize = kekulize

    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        # get molecule
        mol = data[self.molecule_df]

        if self.kekulize:
            Chem.Kekulize(mol)

        # replace molecule
        data[self.molecule_df] = mol

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.molecule_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.molecule_df]
    
    def args_repr(self) -> str:
        return (
            f'kekulize={self.kekulize}'
        )
        

class DFAtomsData(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            result_nodes_df: str,
            atom_types_df: str        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df

        # datafield where to put the resulting node features
        self.result_nodes_df = result_nodes_df

        # datafield from which to get atom types
        self.atom_types_df = atom_types_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        # check if atom types dictionary exists
        if self.atom_types_df not in data:
            data[self.atom_types_df] = dict()

        atom_types = data[self.atom_types_df]

        # get molecule
        mol = data[self.molecule_df]

        # collect atom types as indices
        type_idx = []

        for atom in mol.GetAtoms():
            
            # check if current atom is considered in
            # the overall dataset atom types
            atom_symb = atom.GetSymbol()

            if atom_symb not in atom_types:
                    atom_types[atom_symb] = len(atom_types)
                
                # append atom type as an index
            type_idx.append(atom_types[atom_symb])

        # put resulting list of atoms into the resulting
        # datafield
        data[self.result_nodes_df] = torch.tensor(type_idx)

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.molecule_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.result_nodes_df, self.atom_types_df]
    

class DFBondsData(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            result_edge_index_df: str,
            result_edge_attr_df: str,
            bond_types_df: str,
            aromatic_bond: bool
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df

        # datafield where to put the resulting edge index
        # and edge attributes
        self.result_edge_index_df = result_edge_index_df
        self.result_edge_attr_df = result_edge_attr_df
        self.aromatic_bond = aromatic_bond

        # datafield from which to get bond types
        self.bond_types_df = bond_types_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        # check if bond types dictionary exists
        if self.bond_types_df not in data:
            data[self.bond_types_df] = dict()

        bond_types = data[self.bond_types_df]

        # get molecule
        mol = data[self.molecule_df]

        # collect 
        row, edge_type = [], []

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = str(bond.GetBondType())

            if bond_type not in bond_types:
                bond_types[bond_type] = len(bond_types)

            row.append([start, end])
            edge_type.append(bond_types[bond_type])

        # tranform to tensor
        if len(row) == 0: # if no edges, then set special case
            edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            edge_index = torch.tensor(row, dtype=torch.long).permute(1, 0)
        edge_attr = torch.tensor(edge_type, dtype=torch.long)

        # put resulting list of atoms into the resulting
        # datafield
        data[self.result_edge_index_df] = edge_index
        data[self.result_edge_attr_df] = edge_attr

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.molecule_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.result_edge_index_df, self.result_edge_attr_df]


# ------------------ codice mio ----------------------

class DFAdditionaFeatureEdges(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            attribute_df_edge: str,
            enable: bool=True
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df
        self.attribute_df_edge = attribute_df_edge
        self.enable=enable

        # datafield where to put the resulting edge index
        # and edge attributes
        #self.result_edge_attr_df = result_edge_attr_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        if  self.enable==False:
            data[self.attribute_df]=None
            return data

        # get molecule
        mol = data[self.molecule_df]

        # extract distance vector
        feature=mol.GetProp('Distance')

        # convert from str type to integer
        feature_list=[locale.atof(elem) for elem in feature.split()]
        
        n=len(feature_list)

        feature_matrix=torch.tensor(feature_list, dtype=torch.float32).view(int(math.sqrt(n)),int(math.sqrt(n)))
        
        # add to the datafield
        data[self.attribute_df_edge]=feature_matrix

        return data

from rdkit.Chem import AllChem

class DFAdditionaFeatureEdges_zinc(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            attribute_df_edge: str,
            enable: bool=True
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df
        self.attribute_df_edge = attribute_df_edge
        self.enable=enable


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        if data['value_error']==False:
            return data

        pos=data['pos']
        distance = []
        for atom_1 in range(pos.shape[0]):
            for atom_2 in range(pos.shape[0]):
                    
                distance.append(np.linalg.norm(pos[atom_1,:].numpy()-pos[atom_2,:].numpy()))
            
        n=len(distance)

        feature_matrix=torch.tensor(distance, dtype=torch.float32).view(int(math.sqrt(n)),int(math.sqrt(n)))
            
            # add to the datafield
        data[self.attribute_df_edge]=feature_matrix

        return data
        



class DFAdditionaCoordinate_zinc(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            attribute_df_pos: str,
            enable: bool=True
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df
        self.attribute_df_pos = attribute_df_pos
        self.enable=enable

        # datafield where to put the resulting edge index
        # and edge attributes
        #self.result_edge_attr_df = result_edge_attr_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        if  self.enable==False:
            data[self.attribute_df]=None
            return data

        try:
            # get molecule
            mol = data[self.molecule_df]
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            AllChem.UFFOptimizeMolecule(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xf00d
            mol=Chem.RemoveHs(mol)
            mol_block = Chem.MolToMolBlock(mol)
            
            coordinates = []
            lines = mol_block.splitlines()
            for line in lines[4:4 + mol.GetNumAtoms()]:  # The coordinates start at line 5
                parts = line.split()
                # Extract the x, y, z coordinates (3rd, 4th, and 5th columns)
                x, y, z = map(float, parts[0:3])
                for elem in [x,y,z]:
                    coordinates.append(elem)
            
            n=len(coordinates)

            feature_matrix=torch.tensor(coordinates, dtype=torch.float32).view(int(n/3),3)
            feature_matrix = feature_matrix - torch.mean(feature_matrix, dim=0, keepdim=True)
            
            # add to the datafield
            data[self.attribute_df_pos]=feature_matrix
            data['value_error']=True
            return data
        except ValueError:
            data['value_error']=False
            return data

    
class DFAdditionaCoordinate(DFBaseTransform):

        def __init__(
                self,
                molecule_df: str,
                attribute_df_pos: str,
                enable: bool=True
            ):

            # rdkit molecule datafield
            self.molecule_df = molecule_df
            self.attribute_df_pos = attribute_df_pos
            self.enable=enable

            # datafield where to put the resulting edge index
            # and edge attributes
            # self.result_edge_attr_df = result_edge_attr_df


        def __call__(
                self,
                data: Dict,
            ) -> Dict:

            if  self.enable==False:
                data[self.attribute_df]=None
                return data

            # get molecule
            mol = data[self.molecule_df]

            # extract distance vector
            feature=mol.GetProp('Coordinates')

            # convert from str type to integer
            feature_list=[locale.atof(elem) for elem in feature.split()]
            
            n=len(feature_list)

            feature_matrix=torch.tensor(feature_list, dtype=torch.float32).view(int(n/3), 3)
            
            # to center the data around the origin and remove translational bias 
            feature_matrix = feature_matrix - torch.mean(feature_matrix, dim=0, keepdim=True)

            # add to the datafield
            data[self.attribute_df_pos]=feature_matrix

            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.molecule_df]
        @property
        def output_df_list(self) -> List[str]:
            return [self.attribute_df_pos]

class DFAdditionaFeatureNodes(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            attribute_df_node: str,
            which: list,
            enable: bool=True
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df
        self.attribute_df_node = attribute_df_node
        self.enable=enable
        self.which=which

        # datafield where to put the resulting edge index
        # and edge attributes
        #self.result_edge_attr_df = result_edge_attr_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        if  self.enable==False:
            data[self.attribute_df_node]=None
            return data

        # get molecule
        mol = data[self.molecule_df]


        # this code is useful for compute some atom features
        '''
        for index,f in enumerate(self.which):
            feature=mol.GetProp(f)
            feature_list=[locale.atof(elem) for elem in feature.split()]
            n=len(feature_list)
            if index==0:
                feature_matrix=torch.tensor(feature_list, dtype=torch.float32).view(n)
            else:
                feature_temp=torch.tensor(feature_list, dtype=torch.float32).view(n)
                feature_matrix=torch.cat((feature_matrix, feature_temp), dim=-1)
        '''

        # this serve to compute formal charges 

        # extract distance vector
        feature=mol.GetProp('Formal_Charge')

        # convert from str type to integer
        feature_list=[locale.atof(elem) for elem in feature.split()]
        
        n=len(feature_list)

        feature_matrix=torch.tensor(feature_list, dtype=torch.float32).view(n)

        # add to the datafield
        data[self.attribute_df_node]=feature_matrix

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.molecule_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.attribute_df_node]

class DFAdditionaFeatureNodes_sdf(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            attribute_df_node: str,
            which: list,
            enable: bool=True
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df
        self.attribute_df_node = attribute_df_node
        self.enable=enable
        self.which=which

        # datafield where to put the resulting edge index
        # and edge attributes
        #self.result_edge_attr_df = result_edge_attr_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        if  self.enable==False:
            data[self.attribute_df_node]=None
            return data
        
        if data['value_error']==False:
            return data

        # get molecule
        mol = data[self.molecule_df]
        formal_charge=[]
        for atom in mol.GetAtoms():
            formal_charge.append(atom.GetFormalCharge())
        
        n=len(formal_charge)

        feature_matrix=torch.tensor(formal_charge, dtype=torch.float32).view(n)

        # add to the datafield
        data[self.attribute_df_node]=feature_matrix

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.molecule_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.attribute_df_node]

# ----------------------------------------------------

class DFGraphToMol(DFBaseTransform):
    
        def __init__(
                self,
                graph_df: str,
                result_mol_df: str,
                atom_decoder_df: str,
                bond_decoder_df: str,
                relaxed: bool=False
            ):
    
            # graph datafield
            self.graph_df = graph_df
    
            # datafield where to put the resulting molecule
            self.result_mol_df = result_mol_df
    
            # datafield from which to get atom types
            self.atom_decoder_df = atom_decoder_df
            self.bond_decoder_df = bond_decoder_df
    
            # relaxed conversion
            self.relaxed = relaxed
    
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
    
            # get graph
            graph = data[self.graph_df]
    
            # get decoder
            atom_decoder = data[self.atom_decoder_df]
            bond_decoder = data[self.bond_decoder_df]
    
            # convert graph to molecule
            converter = GraphToMoleculeConverter(
                atom_decoder =  atom_decoder,
                bond_decoder =  bond_decoder,
                relaxed =       self.relaxed
            )
            mol = converter(graph)
    
            # put resulting molecule into the resulting
            # datafield
            data[self.result_mol_df] = mol

            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.graph_df]
        @property
        def output_df_list(self) -> List[str]:
            return [self.result_mol_df]
        
        def args_repr(self) -> str:
            return (
                f'relaxed={self.relaxed}'
            )

class DFMolToSmiles(DFBaseTransform):

    def __init__(
            self,
            mol_df: str,
            smiles_df: str,
            canonical:bool=True, 
            isomeric:bool=False,
            sanitize_smiles: bool=False,
        ):

        self.mol_df = mol_df
        self.smiles_df = smiles_df
        self.sanitize_smiles = sanitize_smiles
        self.canonical = canonical
        self.isomeric = isomeric


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        smiles = mol2smiles(data[self.mol_df], sanitize=self.sanitize_smiles, canonical=self.canonical, isomeric_smile=self.isomeric)
        data[self.smiles_df] = smiles

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.mol_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.smiles_df]
    
    def args_repr(self) -> str:
        return (
            f'sanitize_smiles={self.sanitize_smiles}'
        )
    

class DFSmilesToMol(DFBaseTransform):
    
        def __init__(
                self,
                smiles_df: str,
                mol_df: str,
                sanitize: bool=True
            ):
    
            self.smiles_df = smiles_df
            self.mol_df = mol_df
            self.sanitize = sanitize
    
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
    
            smiles = data[self.smiles_df]
            mol = Chem.MolFromSmiles(str(smiles), sanitize=self.sanitize)

            data[self.mol_df] = mol
    
            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.smiles_df]
        @property
        def output_df_list(self) -> List[str]:
            return [self.mol_df]
        
        def args_repr(self) -> str:
            return (
                f'sanitize={self.sanitize}'
            )
    

class DFMolToNxGraph(DFBaseTransform):
        
        def __init__(
                self,
                mol_df: str,
                nx_df: str,
            ):
    
            self.mol_df = mol_df
            self.nx_df = nx_df
    
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
    
            mol = data[self.mol_df]
    
            graph = mol2nx(mol)
    
            data[self.nx_df] = graph
    
            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.mol_df]
        @property
        def output_df_list(self) -> List[str]:
            return [self.nx_df]

class AssertAtomTypes(DFBaseTransform):
    
    def __init__(
        self,
        atom_types: str,
        remove_hydrogen: bool
    ):
    
        self.atom_types = atom_types
        self.remove_hydrogen = remove_hydrogen
    
    def __call__(
                self,
                data: Dict,
            ) -> Dict:
    
            if self.remove_hydrogen:
                index=data[self.atom_types]['H']
                data[self.atom_types].pop('H')
                for key in data[self.atom_types].keys():
                    if data[self.atom_types][key]>index:
                        data[self.atom_types][key]=data[self.atom_types][key]-1
    
            return data
        
    @property
    def input_df_list(self) -> List[str]:
            return [self.atom_types]
    @property
    def output_df_list(self) -> List[str]:
            return [self.atom_types]
    


class DFToGraph(DFBaseTransform):

    def __init__(
            self,
            *others_df,
            result_df: str,
            nodes_df: str,
            edge_index_df: str,
            edge_attr_df: str,
            targets_df: str,
            atom_types: str,
            remove_hydrogen: bool
        ):

        # datafield where to put the resulting graph
        self.result_df = result_df

        # main graph components
        self.nodes_df = nodes_df
        self.edge_index_df = edge_index_df
        self.edge_attr_df = edge_attr_df
        self.targets_df = targets_df

        # other datafields
        self.others_df = others_df

        self.atom_types= atom_types
        
        self.remove_hydrogen = remove_hydrogen


    def __call__(
            self,
            data: Dict,
        ) -> Dict:
        
        if data['value_error']==False:
            return data

        try: 
            y = data[self.targets_df]
        except KeyError:
            y=1

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float)

        graph = SparseGraph(
            x=			data[self.nodes_df],
            edge_index=	data[self.edge_index_df],
            edge_attr=	data[self.edge_attr_df],
            y=			y,
            **{df: data[df] for df in self.others_df}
        )
        
        data[self.result_df] = graph

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [
            self.nodes_df,
            self.edge_index_df,
            self.edge_attr_df,
            self.targets_df
            ] + list(self.others_df)
    @property
    def output_df_list(self) -> List[str]:
        return [self.result_df]
    

class DFRemoveGraphsInvalidMol(DFBaseTransform):
    def __init__(
            self,
            graph_list_df: str,
            atom_decoder_df: str,
            bond_decoder_df: str=None,
            others_df: List[str]=None
        ):

        self.graph_list_df = graph_list_df
        self.atom_decoder_df = atom_decoder_df
        self.bond_decoder_df = bond_decoder_df
        self.others_df = others_df if others_df is not None else []

    
    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        converter = GraphToMoleculeConverter(
            atom_decoder =  data[self.atom_decoder_df],
            bond_decoder =  data[self.bond_decoder_df] if self.bond_decoder_df is not None else None,
            relaxed =       False
        )

        graph_list = data[self.graph_list_df]

        invalid = 0
        total = len(graph_list)

        pbar = tqdm(enumerate(graph_list), total=total)

        for i, graph in pbar:
            mol = converter(graph)
            smiles = mol2smiles(mol)

            if smiles is None or smiles == '':
                # remove graph
                for df in [self.graph_list_df] + self.others_df:
                    del data[df][i]
                invalid += 1

            pbar.set_postfix({'invalid': f'{invalid}/{total}'})

        return data
    

class DFComputeMolecularProperties(DFBaseTransform):
    def __init__(
            self,
            mol_df: str,
            graph_df: str,
            properties: List[str],
        ):

        self.mol_df = mol_df
        self.graph_df = graph_df
        self.properties = properties

    
    def __call__(
            self,
            data: Dict,
        ) -> Dict:
        
        if data['value_error']==False:
            return data

        mol = data[self.mol_df]
        prop_values = []

        for prop in self.properties:
            func = PROPS_NAME_TO_FUNC[prop]
            prop_values.append(func(mol))

        prop_values = torch.tensor(prop_values, dtype=torch.float)

        if hasattr(data[self.graph_df], 'y') and data[self.graph_df].y is not None:
            y = torch.cat([data[self.graph_df].y, prop_values], dim=-1)
        else:
            y = prop_values

        data[self.graph_df].y = y

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.mol_df, self.graph_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.graph_df]
    
    def args_repr(self) -> str:
        return (
            f'properties={self.properties}'
        )
    

class CondMoleculeHasValidSMILES(Condition):
    def __init__(
            self,
            obj_to_check_df: str
        ):
        
        self.obj_to_check_df = obj_to_check_df


    def __call__(self, data: Dict) -> bool:
        
        # check if current molecule has a valid SMILES
        mol = data[self.obj_to_check_df]
        smiles = Chem.MolToSmiles(mol, canonical=True)

        return smiles is not None and smiles != ''
        
class CondValidSMILES(Condition):
    def __init__(
            self,
            obj_to_check_df: str
        ):
        
        self.obj_to_check_df = obj_to_check_df


    def __call__(self, data: Dict) -> bool:
        smiles = data[self.obj_to_check_df]

        return smiles is not None and smiles != ''

class DFAddFeaturesSDF_zinc(DFBaseTransform):
    def __init__(
            self,
            datafield: str,
            remove_hs: bool
    ):
        self.file_data=datafield
        self.remove_hs=remove_hs

    def __call__(self,data):

        filename: str = data[self.file_data]
        print('Starting Calculating Chemical Features ...')

        if not self.remove_hs:
            if os.path.isfile('datasets/zinc/raw/zinc_feat_hs.sdf'):
                print('Dataset alreay transformed.')
                data['mols_file']='datasets/zinc/raw/zinc_feat_hs.sdf'
                return data
        else:
             if os.path.isfile('datasets/zinc/raw/zinc_feat.sdf'):
                print('Dataset alreay transformed.')
                data['mols_file']='datasets/zinc/raw/zinc_feat.sdf'
                return data
        if not self.remove_hs:
            suppl=Chem.SDMolSupplier(filename, sanitize=False, removeHs=False)
            w=Chem.SDWriter(data['root_path']+"/raw/zinc_feat_hs.sdf")
            mols = [mol for mol in suppl if mol is not None]
        else:
            suppl=Chem.SDMolSupplier(filename, sanitize=False)
            w=Chem.SDWriter(data['root_path']+"/raw/zinc_feat.sdf")
            mols = [mol for mol in suppl if mol is not None]
            mols = [Chem.RemoveHs(mol, sanitize=False) for mol in suppl if mol is not None]

        # compute the distances for each couple of atoms
        for mol in tqdm(mols):
            # compute distance between atoms 
            mol.GetConformer()
            distance=[]
            formal_charge=[]
            coordinates=[]

            for i,atom in enumerate(mol.GetAtoms()):

                # single first atom position
                pos_i = mol.GetConformer().GetAtomPosition(i)
                formal_charge.append(atom.GetFormalCharge())

                # coordinate of first atom
                point_i=np.array([pos_i.x, pos_i.y, pos_i.z])
                if len(coordinates)==0:
                    coordinates=point_i
                else:
                    coordinates = np.concatenate((coordinates, point_i))
                for j,atom in enumerate(mol.GetAtoms()):
                    pos_j = mol.GetConformer().GetAtomPosition(j)
                    point_j=np.array([pos_j.x, pos_j.y, pos_j.z])
                    distance.append(np.linalg.norm(point_i-point_j))


            # convert all the quantity to string, necessary for writing in sdf file
            distance_str = ' '.join(map(str, distance))
            formal_charge_str = ' '.join(map(str, formal_charge))
            coordinates_str = ' '.join(map(str, coordinates))

            mol.SetProp("Distance", distance_str)
            mol.SetProp("Formal_Charge", formal_charge_str)
            mol.SetProp('Coordinates', coordinates_str)
        
        
        # save modifications
        for mol in mols:
            w.write(mol)
        w.close()

        print('End Calculations')

        # change data values
        if not self.remove_hs:
            data['mols_file']='datasets/zinc/raw/zinc_feat_hs.sdf'
        else:
            data['mols_file']='datasets/zinc/raw/zinc_feat.sdf'

        return data