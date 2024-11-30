# runtime transforms
from torch_geometric.transforms import BaseTransform, Compose
from src.data.pipelines.runtime.core import MyToUndirected

from src.noise.data_transform.subsequence_sampler import SubgraphSequenceSampler
from src.noise.config_support import build_noise_process
from src.noise.removal import (
    resolve_removal_process,
    resolve_removal_schedule
)

from src.noise.timesample import resolve_timesampler


from src.data.pipelines import reg_runtime_t


class PrepareData(BaseTransform):
    """transform for preparing data for removal process"""
    def __init__(self, removal_process):

        self.removal_process = removal_process

    def __call__(self, batch):
            
        self.removal_process.prepare_data(datapoint=batch)
        return batch


@reg_runtime_t.register('ifh')
class IFHTransform:

    def __init__(self):
        pass

    def __repr__(self):
        return 'IFHTransform()'

    def __call__(self, removal, **kwargs):

        removal_process, timesampler = build_noise_process(
            removal,
            resolve_removal_process,
            resolve_removal_schedule,
            resolve_timesampler
        )

        pipeline = Compose([
            # transform graph into undirected graph
            MyToUndirected(),
            # prepare data for removal process
            # e.g., compute ordering during dataloading on cpu
            PrepareData(removal_process)
        ])

        return pipeline