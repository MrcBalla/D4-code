from src.utils.decorators import InstanceRegister

reg_download = InstanceRegister('Download')
reg_preprocess = InstanceRegister('Preprocess')
reg_runtime_t = InstanceRegister('RuntimeTransforms')
reg_postprocess = InstanceRegister('Postprocess')

import pkgutil

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    loader.find_module(module_name).load_module(module_name)