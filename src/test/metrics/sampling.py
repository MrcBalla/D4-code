from typing import Dict, List, Tuple, Any, Optional, Union, Callable

import numpy as np
import torch
from torch import nn, Tensor
import re
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Geometry import Point3D
from collections import Counter
import os

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from omegaconf import OmegaConf, DictConfig
import networkx as nx

from src.data.simple_transforms.molecular import mol2smiles, GraphToMoleculeConverter
from src.data.datamodule import GraphDataModule

import src.test.metrics as m_list
import src.test.utils.molecular as molecular
import src.test.utils.synth as synth
import src.test.utils.graphgdp_metrics.evaluator as graphgdp_metrics

from src.test import reg_metrics

from src.datatypes.dense import to_dense_adj_bipartite
from scipy.stats import kstest
import math

from rdkit.Chem import AllChem
import copy

# the code for computing the Wasserstein distance is taken from MiDi code https://github.com/cvignac/MiDi/tree/master

class BaseSamplingMetric(nn.Module):

    def __init__(self):
        super().__init__()

    
    def override(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        return self


    def __call__(self, **kwargs) -> Dict:
        raise NotImplementedError


def is_sampling_metric(metric: Callable) -> bool:
    return isinstance(metric, BaseSamplingMetric)

def contains_sampling_metrics(metrics: Dict[str, Callable]) -> bool:
    return any([is_sampling_metric(metric) for metric in metrics.values()])
 

################################################################################
#                          MOLECULAR SAMPLING METRICS                          #
################################################################################


@reg_metrics.register(m_list.KEY_BOND_DISTANCE_METRIC)
class bond_distance_distribution(BaseSamplingMetric):
    def __init__(self, test_mol:  List[str] = None):
        super().__init__()
        self.test_mol = test_mol
        
    def compute_distance_diff(
        self,
        generated_graphs: List[Data]
    ):

        sum_t_ = {0: 0, 1: 0, 2: 0, 3:0}
        for t in self.test_mol:
            elem = torch.argmax(t.edge_attr,1)
            for e in elem.tolist():
                sum_t_[e]+=1

        generated_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}

        generated_dist_to_save = []
        generated_dist_to_save_not_bonded = []
        bond_type_s = []

        atom_type = []

        for mol in generated_graphs:
                    if len(atom_type)==0:
                        atom_type=mol.x.tolist()
                    else:
                        atom_type=atom_type+mol.x.tolist()
                        
                    dist_vector_1 =mol.attribute_edge.reshape(mol.x.shape[0], mol.x.shape[0])
                    dist_vector_1 = dist_vector_1[dist_vector_1!=0]
                    distances_to_consider_1 = torch.round(dist_vector_1, decimals=2)

                    edge_tensor = (to_dense_adj_bipartite(mol.edge_index, mol.edge_attr+1,
                                                            max_num_nodes_s = mol.x.shape[0], max_num_nodes_t =mol.x.shape[0])!=0).squeeze(0)
                    dist_vector = torch.mul(mol.attribute_edge.reshape(mol.x.shape[0], mol.x.shape[0]), (edge_tensor!=0).int()).squeeze(0)
                    dist_vector = dist_vector[dist_vector!=0]

                    distances_to_consider = torch.round(dist_vector, decimals=2)
                    if len(generated_dist_to_save)==0:
                        generated_dist_to_save=distances_to_consider.tolist()
                        generated_dist_to_save_not_bonded=distances_to_consider_1.tolist()
                        bond_type_s = (mol.edge_attr+1).tolist()
                    else:
                        generated_dist_to_save=generated_dist_to_save+distances_to_consider.tolist()
                        generated_dist_to_save_not_bonded = generated_dist_to_save_not_bonded+distances_to_consider_1.tolist()
                        bond_type_s = bond_type_s + (mol.edge_attr+1).tolist()

                    for i, d in enumerate(distances_to_consider):
                        generated_bond_lenghts[mol.edge_attr[i].item()+1][d.item()] += 1


        for bond_type in range(1,5):
            s = sum(generated_bond_lenghts[bond_type].values())
            if s == 0:
                s = 1
            for d, count in generated_bond_lenghts[bond_type].items():
                generated_bond_lenghts[bond_type][d] = count / s

        target = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
        for test in self.test_mol:
            edge_tensor = (to_dense_adj_bipartite(test.edge_index, torch.argmax(test.edge_attr,1)+1)!=0)
            dist = test.attribute_edge[:(test.x.shape[0]),:(test.x.shape[0])]
            dist_vector = torch.mul(dist, (edge_tensor!=0).int()).squeeze(0)
            dist_vector = dist_vector[dist_vector!=0]
            distances_to_consider = torch.round(dist_vector, decimals=2)
            for i, d in enumerate(distances_to_consider):
                    target[torch.argmax(test.edge_attr,1)[i].item()+1][d.item()] += 1

        for bond_type in range(1,5):
            s = sum(target[bond_type].values())
            if s == 0:
                s = 1
            for d, count in target[bond_type].items():
                target[bond_type][d] = count / s


        min_generated_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in generated_bond_lenghts.values())
        min_target_length = min(min(d.keys()) if len(d) > 0 else 1e4 for d in target.values())
        min_length = min(min_generated_length, min_target_length)

        max_generated_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in generated_bond_lenghts.values())
        max_target_length = max(max(bl.keys()) if len(bl) > 0 else -1 for bl in target.values())
        max_length = max(max_generated_length, max_target_length)

        num_bins = int((max_length - min_length) * 100) + 1
        generated_bond_lengths = torch.zeros(4, num_bins)
        target_bond_lengths = torch.zeros(4, num_bins)

        for bond_type in range(1,5):
            for d, count in generated_bond_lenghts[bond_type].items():
                bin = int((d - min_length) * 100)
                generated_bond_lengths[bond_type - 1, bin] = count
            for d, count in target[bond_type].items():
                bin = int((d - min_length) * 100)
                target_bond_lengths[bond_type - 1, bin] = count

        cs_generated = torch.cumsum(generated_bond_lengths, dim=1)
        cs_target = torch.cumsum(target_bond_lengths, dim=1)

        w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1) / 100    # 100 because of bin size

        # this part of code allow to understand which dataset are we evaluating:
        # graphs with on average more bonds are related to gdb13, fewer bonds to qm9
        total_len = []
        for graph in generated_graphs:
            total_len.append(len(graph.edge_attr))

        if np.average(total_len)/len(generated_graphs)>=24:
            return torch.sum(w1_per_class*torch.tensor([0.1547,0.0255,0.003,0]))  # these values are chosen equally to MiDi paper to be reproducible, these represent the frequency of bonds considering also no bonds type
        else:    
            return torch.sum(w1_per_class*torch.tensor([0.0079, 0.0273, 0.2388, 0])) # these values are chosen equally to MiDi paper to be reproducible, these represent the frequency of bonds considering also no bonds type
    
    def __call__(self, data: List[Data]):
        w1 = self.compute_distance_diff(data)
        ret = {
            m_list.KEY_BOND_DISTANCE_METRIC: w1,
        }
        ret['bond_distance_metric'] = w1.mean()
        
        return ret

@reg_metrics.register(m_list.KEY_MOLECULAR_VALIDITY)
class ValidMoleculeMetric(BaseSamplingMetric):
    
    def __init__(self, graph_to_mol_converter: GraphToMoleculeConverter=None, relaxed: bool=True, ret_valid: bool=False, ret_conn_comps: bool=False):
        super().__init__()

        self.graph_to_mol_converter = graph_to_mol_converter
        self.relaxed = relaxed
        self.ret_valid = ret_valid
        self.ret_conn_comps = ret_conn_comps

    def compute_validity(
            self,
            generated_graphs: List[Data]

        ) -> Tuple[List[str], float, np.ndarray, List[str]]:
        """ generated: list of couples (positions, atom_types)"""
        
        valid_smiles = []
        num_components = []
        all_smiles = []
        error_message = Counter()

        for graph in generated_graphs:
            
            # torch_geometric graph to RDKit molecule
            mol = self.graph_to_mol_converter(
                graph,
                override_relaxed=self.relaxed
            )
                
            # RDKit molecule to string (SMILES)
            if mol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                    num_components.append(len(mol_frags))
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol)
                    valid_smiles.append(smiles)
                    all_smiles.append(smiles)
        
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    # print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    # print("Can't kekulize molecule")
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
        print(f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
              f" -- No error {error_message[-1]}")
        
        return valid_smiles, len(valid_smiles) / len(generated_graphs), np.array(num_components), all_smiles
    

    def __call__(self, data: List[Data]):
        valid_smiles, validity, num_components, all_smiles = self.compute_validity(data)
        ret = {
            m_list.KEY_MOLECULAR_VALIDITY: validity,
        }
        if self.ret_valid:
            ret['valid_smiles'] = valid_smiles
        if self.ret_conn_comps:
            ret['num_components'] = dict(
                min=num_components.min(),
                max=num_components.max(),
                mean=num_components.mean()
            )

        return ret


@reg_metrics.register(m_list.KEY_MOLECULAR_UNIQUENESS)
class UniqueMoleculeMetric(BaseSamplingMetric):
        
    def __init__(self, ret_unique: bool=False):
        super().__init__()

        self.ret_unique = ret_unique

    
    def compute_uniqueness(
        self,
        valid_smiles: List[str]
    ) -> Tuple[List[str], float]:
        if len(valid_smiles) == 0: return [], 0

        return list(set(valid_smiles)), len(set(valid_smiles)) / len(valid_smiles)


    def __call__(self, smiles: List[str]):
        unique_smiles, uniqueness = self.compute_uniqueness(smiles)

        ret = {
            m_list.KEY_MOLECULAR_UNIQUENESS: uniqueness,
        }
        if self.ret_unique:
            ret['unique_smiles'] = unique_smiles
        
        return ret


@reg_metrics.register(m_list.KEY_MOLECULAR_NOVELTY)
class NovelMoleculeMetric(BaseSamplingMetric):
            
    def __init__(self, ref_smiles: List[str] = None, ret_novel: bool=False):
        super().__init__()

        self.ret_novel = ret_novel
        self.ref_smiles = ref_smiles

    
    def compute_novelty(
        self,
        unique_smiles: List[str],
    ) -> Tuple[List[str], float]:

        if len(unique_smiles) == 0: return [], 0
        num_novel = 0
        novel = []
        if self.ref_smiles is None:
            print("Dataset smiles is None, novelty computation skipped")
            return [], 1
        for smiles in unique_smiles:
            if smiles not in self.ref_smiles:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique_smiles)


    def __call__(
            self,
            smiles: List[str]
        ):
        novel_smiles, novelty = self.compute_novelty(smiles)

        ret = {
            m_list.KEY_MOLECULAR_NOVELTY: novelty,
        }
        if self.ret_novel:
            ret['novel_smiles'] = novel_smiles
        
        return ret

@reg_metrics.register(m_list.KEY_FCD)
class FCDMetric(BaseSamplingMetric):
        
    def __init__(self, ref_smiles = None, n_jobs: int = 1, device: Optional[str]=None, batch_size: int=512):
        super().__init__()

        self.n_jobs = n_jobs
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.batch_size = batch_size

        self.fcd = None
        if ref_smiles is not None:
            self.precompute_intermediate_FCD(ref_smiles)


    def precompute_intermediate_FCD(
            self,
            ref_smiles: List[str]
        ):

        self.fcd = molecular.get_FCDMetric(
            ref_smiles = ref_smiles,
            n_jobs = self.n_jobs,
            device = self.device,
            batch_size = self.batch_size
        )

        return self


    def __call__(
            self,
            smiles: List[str],
            ref_smiles: Optional[List[str]]=None
        ):

        if ref_smiles is not None:
            self.precompute_intermediate_FCD(ref_smiles)

        if self.fcd is None:
            raise ValueError("FCDMetric: precompute intermediate FCD not called")
        
        return {m_list.KEY_FCD: self.fcd(smiles)}


@reg_metrics.register(m_list.KEY_NSPDK)
class NSPDKMetric(BaseSamplingMetric):
            
        def __init__(self, ref_nx = None, n_jobs: int = 10):
            super().__init__()
    
            self.n_jobs = n_jobs
            self.ref_nx = ref_nx
    
    
        def __call__(
                self,
                graphs: List[nx.Graph]
            ):

            nspdk = molecular.compute_nspdk_mmd(graphs, self.ref_nx, metric='nspdk', is_hist=False, n_jobs=self.n_jobs)
    
            return {m_list.KEY_NSPDK: nspdk}



################################################################################
#                      GRAPH GENERATION SAMPLING METRICS                       #
################################################################################

from networkx import number_connected_components

@reg_metrics.register(m_list.KEY_GRAPH_CONN_COMP)
class GraphConnCompMetric(BaseSamplingMetric):

    def __init__(self):
        super().__init__()


    def __call__(self, generated_graphs: List[nx.Graph]) -> Dict:
        
        conn_comps = [number_connected_components(g) for g in generated_graphs]

        conn_comps = dict(
            min =   np.min(conn_comps),
            max =   np.max(conn_comps),
            mean =  np.mean(conn_comps)
        )

        return {m_list.KEY_GRAPH_CONN_COMP: conn_comps}
        


class GraphMMDMetric(BaseSamplingMetric):

    def __init__(
            self,
            metric: Callable,
            name: str,
            test_graphs: List[nx.Graph] = None,
            compute_emd: bool = True,
            **kwargs
        ):
        super().__init__()

        self.test_graphs = test_graphs
        self.metric = metric
        self.name = name
        self.compute_emd = compute_emd
        self.kwargs = kwargs

    def __call__(self, generated_graphs: List) -> Dict:
        value = self.metric(
            self.test_graphs,
            generated_graphs,
            compute_emd=self.compute_emd,
        )
        return {self.name: value}

        
@reg_metrics.register(m_list.KEY_GRAPH_DEGREE)
class DegreeMetric(GraphMMDMetric):
    
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.degree_stats,
            name = m_list.KEY_GRAPH_DEGREE,
            compute_emd = compute_emd,
            is_parallel=True
        )

@reg_metrics.register(m_list.KEY_GRAPH_SPECTRE)
class SpectreMetric(GraphMMDMetric):
        
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.spectral_stats,
            name = m_list.KEY_GRAPH_SPECTRE,
            compute_emd = compute_emd,
            is_parallel=True,
            n_eigvals=-1
        )

@reg_metrics.register(m_list.KEY_GRAPH_CLUSTERING)
class ClusteringMetric(GraphMMDMetric):
            
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.clustering_stats,
            name = m_list.KEY_GRAPH_CLUSTERING,
            compute_emd = compute_emd,
            is_parallel=True,
            bins=100
        )

@reg_metrics.register(m_list.KEY_GRAPH_ORBIT)
class OrbitMetric(GraphMMDMetric):
                
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.orbit_stats_all,
            name = m_list.KEY_GRAPH_ORBIT,
            compute_emd = compute_emd
        )


@reg_metrics.register(m_list.KEY_GRAPH_GIN)
class GraphGinMetric(BaseSamplingMetric):
                    
    def __init__(
            self,
            test_graphs: List[nx.Graph] = None,
            cfg = None
        ):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.create({})

        self.fn = graphgdp_metrics.get_nn_eval(cfg)
        self.test_graphs = test_graphs
        

    def __call__(self, generated_graphs: List) -> Dict:
        values = self.fn(
            test_dataset = self.test_graphs,
            pred_graph_list = generated_graphs
        )
        value = values['gin_MMD_RBF_mean']
        return {m_list.KEY_GRAPH_GIN: value}
    

################################################################################
#                           PROCESS SAMPLING METRICS                           #
################################################################################

@reg_metrics.register(m_list.KEY_SAMPLING_TIME)
class SamplingTimeMetric(BaseSamplingMetric):
    
        def __init__(self):
            super().__init__()
    
        def __call__(self, time) -> Dict:
            return {m_list.KEY_SAMPLING_TIME: time['end'] - time['start']}
        

@reg_metrics.register(m_list.KEY_SAMPLING_MEMORY)
class SamplingMemoryMetric(BaseSamplingMetric):
            
        def __init__(self):
            super().__init__()
    
        def __call__(self, memory) -> Dict:
            return {m_list.KEY_SAMPLING_MEMORY: memory['peak']}