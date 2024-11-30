

from src.data.pipelines import reg_postprocess
from src.data.simple_transforms.graph import GraphToNetworkxConverter

@reg_postprocess.register('networkx')
class NetworkxPostprocess:
    def __init__(
            self,
            to_undirected: bool = False,
            remove_self_loops: bool = False
        ):
        self.to_undirected = to_undirected
        self.remove_self_loops = remove_self_loops


    def __call__(self, dataset_info, **kwargs):
        converter = GraphToNetworkxConverter(
            to_undirected = self.to_undirected,
            remove_self_loops = self.remove_self_loops
        )

        return converter