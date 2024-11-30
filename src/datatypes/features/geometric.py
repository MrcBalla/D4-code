
import torch
from torch_geometric.data import Data
import time

from src.datatypes.features.core import Feature
from src.datatypes.dense import DenseGraph
from src.datatypes.sparse import SparseGraph
from rdkit import Chem
import numpy as np
import math
from rdkit.Chem import AllChem
from src.datatypes.features import reg_features
from src.datatypes.dense import dense_remove_self_loops
from src.datatypes.dense import  remove_no_edge
from src.data.simple_transforms.molecular import GraphToMoleculeConverter
from src.datatypes.dense import dense_graph_to_sparse_graph
from src.data.simple_transforms.molecular import mol2smiles

import src.data.transforms.molecular as prep_tool



@reg_features.register('indegree')
class InDegreeFeature(Feature):

    def get_added_dims(self):
        return {'x': 1}

    def __call__(self, graph: Data) -> Data:

        if isinstance(graph, (SparseGraph, DenseGraph)):
            indegree = graph.indegree
        elif isinstance(graph, (tuple, list)):
            graph, graph_outgoing_edges = graph
            # in undirected graph, remv -> surv is the same as surv -> remv
            # so outdegree of surv is the same as indegree of remv
            if graph_outgoing_edges is None:
                add_indegree = 0
            else:
                add_indegree = graph_outgoing_edges.outdegree
            indegree = graph.indegree + add_indegree
        
        graph.x = torch.cat([graph.x, indegree.unsqueeze(-1)], dim=-1)


@reg_features.register('nodes_num')
class NodesNumFeature(Feature):

    def get_added_dims(self):
        return {'y': 1}

    def __call__(self, graph: Data) -> Data:
        if isinstance(graph, (SparseGraph, DenseGraph)):
            graph = graph
        elif isinstance(graph, (tuple, list)):
            graph = graph[0]

        nodes_num = graph.num_nodes_per_sample
        graph.y = torch.cat([graph.y, nodes_num.unsqueeze(-1)], dim=-1)

        return graph
    
def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


@reg_features.register('eigen_distance')
class eigen_distance(Feature):

    def get_added_dims(self):
        return {'dist': 1}

    def __call__(self, graph: Data) -> Data:
        if isinstance(graph, (SparseGraph, DenseGraph)):
            graph = graph
        elif isinstance(graph, (tuple, list)):
            graph = graph[0]
        
        A = graph.edge_adjmat[..., 1:].sum(dim=-1).float() * graph.node_mask.unsqueeze(1) * graph.node_mask.unsqueeze(2)
        diag = torch.sum(A, dim=-1)
        n = diag.shape[-1]
        D = torch.diag_embed(diag)
        combinatorial = D - A
        diag0 = diag.clone()
        diag[diag == 0] = 1e-12
        diag_norm = 1 / torch.sqrt(diag)
        D_norm = torch.diag_embed(diag_norm) 
        L = torch.eye(n).unsqueeze(0).to(graph.x.device) - D_norm @ A @ D_norm
        L[diag0 == 0] = 0
        (L + L.transpose(1, 2)) / 2
        
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1], device=L.device).unsqueeze(0)
        mask_diag = mask_diag * (~graph.node_mask.unsqueeze(1)) * (~graph.node_mask.unsqueeze(2))
        L = L * graph.node_mask.unsqueeze(1) * graph.node_mask.unsqueeze(2) + mask_diag
        
        eigvals, eigvectors = torch.linalg.eigh(L)
        eigvals = eigvals / torch.sum(graph.node_mask, dim=1, keepdim=True)
        eigvectors = eigvectors * graph.node_mask.unsqueeze(2) * graph.node_mask.unsqueeze(1)
        '''
        n_connected = (eigvals < 1e-5).sum(dim=-1)
        
        k=5
        
        # Get the eigenvectors corresponding to the first nonzero eigenvalues
        to_extend = max(n_connected) + k - n
                             # bs, n, k
        first_k_ev = torch.gather(eigvectors, dim=2, 
                                  index=(torch.arange(k, device=eigvectors.device).repeat(32).reshape(32,5)+n_connected.unsqueeze(1)).unsqueeze(1).expand(-1,n,-1))       # bs, n, k
        first_k_ev = first_k_ev * graph.node_mask.unsqueeze(2)

        graph.eigen=graph.first_k_ev
        '''
        graph.attribute_edge=torch.concat([graph.attribute_edge.unsqueeze(1), eigvectors.unsqueeze(1)], dim=1)
        graph.attribute_edge=graph.attribute_edge.permute(0,2,3,1)
        return graph
    
    


