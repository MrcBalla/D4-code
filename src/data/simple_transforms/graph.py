from typing import Union, List

import networkx as nx

from torch_geometric.utils.convert import (
    to_networkx,
    from_networkx
)

from src.data.simple_transforms import batched
from src.datatypes.sparse import SparseGraph


@batched
def graph2nx(graph, to_undirected=False, remove_self_loops=False, **kwargs):
    return to_networkx(
        graph,
        to_undirected =     to_undirected,
        remove_self_loops = remove_self_loops,
        **kwargs
    )


@batched
def nx2graph(nx_graph, **kwargs):
    return from_networkx(
        nx_graph,
        **kwargs
    )


class GraphToNetworkxConverter:

    def __init__(
            self,
            to_undirected: bool = False,
            remove_self_loops: bool = False
        ):
        self.to_undirected = to_undirected
        self.remove_self_loops = remove_self_loops

    def __call__(
            self,
            batch: Union[List[SparseGraph], SparseGraph],
            **kwargs
        ) -> Union[List[nx.Graph], nx.Graph]:
        return graph2nx(
            batch,
            to_undirected =     self.to_undirected,
            remove_self_loops = self.remove_self_loops,
            **kwargs
        )
    
    def graph_to_nx(self, graph: SparseGraph, **kwargs) -> nx.Graph:
        return self(
            graph,
            to_undirected =     self.to_undirected,
            remove_self_loops = self.remove_self_loops,
            **kwargs
        )
    
    def nx_to_graph(self, nx_graph: nx.Graph, **kwargs) -> SparseGraph:
        return nx2graph(
            nx_graph,
            **kwargs
        )