from src.utils.decorators import ClassRegister

reg_models = ClassRegister('Model')
reg_architectures = ClassRegister('Architecture')

# import all models module file programmatically
import pkgutil

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    loader.find_module(module_name).load_module(module_name)