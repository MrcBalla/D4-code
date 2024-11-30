from typing import List
from src.utils.decorators import ClassRegister

reg_features = ClassRegister('Feature')


def get_features_list(f_names: List[str]):
    return [reg_features.get_instance(f) for f in f_names]


import src.datatypes.features.geometric