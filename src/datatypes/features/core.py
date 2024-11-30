from abc import ABC, abstractmethod
from typing import Dict, List

from torch_geometric.data import Data

from copy import deepcopy


class FeatureException(Exception):
    pass


class Feature:

    def apply_added_dims(self, dims: Dict) -> Dict:
        return increase_dims(dims, increase=self.get_added_dims())


    @abstractmethod
    def get_added_dims(self) -> Dict:
        return {}
    


def increase_dims(dims: Dict, increase: Dict) -> Dict:
    dims_copy = deepcopy(dims)

    for k, v in increase.items():
        if k not in dims_copy:
            raise FeatureException(f"Dimension {k} not found in dims when trying to increase it.")
        dims_copy[k] += v

    return dims_copy


def increase_dims_list(dims: Dict, features: List[Feature]) -> Dict:
    dims_copy = deepcopy(dims)

    for f in features:
        dims_copy = f.apply_added_dims(dims_copy)

    return dims_copy