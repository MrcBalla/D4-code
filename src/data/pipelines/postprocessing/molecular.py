

from src.data.pipelines import reg_postprocess
from src.data.simple_transforms.molecular import GraphToMoleculeConverter

@reg_postprocess.register('molecular')
class MolecularPostprocess:
    def __init__(
            self,
            relaxed: bool = True,
            post_hoc_mols_fix: bool = False,
            post_hoc_mols_convert: bool = False
        ):
        self.relaxed = relaxed
        self.post_hoc_mols_fix = post_hoc_mols_fix
        self.post_hoc_mols_convert = post_hoc_mols_convert

    def __call__(self, dataset_info, **kwargs):
        converter = GraphToMoleculeConverter(
            atom_decoder =          dataset_info['atom_types'],
            bond_decoder =          dataset_info['bond_types'],
            relaxed =               self.relaxed,
            post_hoc_mols_fix =     self.post_hoc_mols_fix,
            post_hoc_mols_convert = self.post_hoc_mols_convert
        )

        return converter