from typing import Dict

import json


from src.data.datamodule import GraphDataModule
import src.data.transforms as base_t
from src.data.transforms import (
    conditions as cond_t,
    molecular as chem_t,
    assertions as assert_t
)
from src.data.simple_transforms.molecular import mol2nx, smiles2mol

from src.data.simple_transforms.molecular import GraphToMoleculeConverter


from src.test import reg_assignment
from src.test.assignment import Assignment
from src.test.metrics.sampling import contains_sampling_metrics


def vun_pipeline(metrics, data_df, torch_to_mol, relaxed, ref_smiles, test_smile):

    compose = base_t.DFCompose([
        # compute validity and return valid smiles
        base_t.DFCustomTransform(
            free_transform=metrics['molecular_validity'].override(
                graph_to_mol_converter =    torch_to_mol,
                relaxed =                   relaxed,
                ret_valid =                 True
            ),
            src_df=data_df,
            dst_df={
                'valid_smiles': 'valid_smiles',
                'molecular_validity': 'molecular_validity'
            }
        ),
        # compute uniqueness and return unique smiles
        base_t.DFCustomTransform(
            free_transform=metrics['molecular_uniqueness'].override(
                ret_unique = True
            ),
            src_df='valid_smiles',
            dst_df={
                'unique_smiles': 'unique_smiles',
                'molecular_uniqueness': 'molecular_uniqueness'
            }
        ),
        # compute novelty
        base_t.DFCustomTransform(
            free_transform=metrics['molecular_novelty'].override(
                ref_smiles = ref_smiles
            ),
            src_df='unique_smiles',
            dst_df={
                'molecular_novelty': 'molecular_novelty'
            }
        ),
        base_t.DFCustomTransform(
            free_transform=metrics['bond_distance_metric'].override(
                test_mol = test_smile
            ),
            src_df=data_df,
            dst_df={
                'bond_distance_metric': 'bond_distance_metric'
            }
        )
    ])

    return compose


def fcd_pipeline(metrics, mols_df, smiles_df, eval_smiles):

    compose = base_t.DFCompose([
        # convert molecules to smiles
        chem_t.DFMolToSmiles(
            mol_df =    mols_df,
            smiles_df =    smiles_df
        ),
        # compute FCD
        base_t.DFCustomTransform(
            free_transform = metrics['fcd'].precompute_intermediate_FCD(
                ref_smiles = eval_smiles,
            ),
            src_df={smiles_df: 'smiles'},
            dst_df={'fcd': 'fcd'}
        )
    ])
    return compose

def nspdk_pipeline(metrics, mols_df, nx_df, eval_nx):

    compose = base_t.DFCompose([
        # convert molecules to nx
        chem_t.DFMolToNxGraph(
            mol_df =   mols_df,
            nx_df =    nx_df
        ),
        # compute nspdk
        base_t.DFCustomTransform(
            free_transform = metrics['nspdk'].override(
                ref_nx = eval_nx
            ),
            src_df={nx_df: 'graphs'},
            dst_df={'nspdk': 'nspdk'}
        )
    ])
    return compose



@reg_assignment.register('molecular')
class MolecularAssignment(Assignment):

    def __init__(
            self,
            metrics_list: Dict[str, Dict],
            split: str,
            datamodule: GraphDataModule,
            how_many_to_generate: int,
            relaxed: bool = True
        ):
        super().__init__(metrics_list)

        self.how_many_to_generate = how_many_to_generate

        # novelty requires training smiles
        if 'molecular_novelty' in self.metrics_list:
            self.train_smiles = datamodule.load_file('train', 'smiles_file_train', json.load)

        # fcd requires evaluation smiles
        if 'fcd' in self.metrics_list:
            self.eval_smiles = datamodule.load_file(split, f'smiles_file_{split}', json.load)

        # nspdk requires evaluation nx graphs
        if 'nspdk' in self.metrics_list:
            if not hasattr(self, 'eval_smiles'):
                self.eval_smiles = datamodule.load_file(split, f'smiles_file_{split}', json.load)

            self.eval_nx = mol2nx(smiles2mol(self.eval_smiles))

        if 'bond_distance_metric' in self.metrics_list:
            self.test_smiles = datamodule.test_dataset()


        # create molecule converter
        dataset_info = datamodule.get_info('train')

        self.torch_to_mol = GraphToMoleculeConverter(
            atom_decoder = dataset_info['atom_types'],
            bond_decoder = dataset_info['bond_types'],
            relaxed = relaxed
        )


        # create pipeline
        self.pipeline = base_t.DFPipeline(

            input_dfs = ['data'],
            output_dfs = self.metrics_list,

            transforms = [

                # check for size if metrics contains sampling metrics
                # base_t.guarded_include(
                #     if_ = contains_sampling_metrics(self.metrics),
                #     do_ = base_t.DFCompose([
                #         base_t.DFCustomTransform(len, 'data', 'data_len'),
                #         assert_t.AssertEqual(
                #             'data_len', value=how_many_to_generate
                #         )
                #     ])
                # ),

                # compute validity, uniqueness and novelty if needed
                vun_pipeline(
                    metrics = self.metrics,
                    data_df = 'data',
                    torch_to_mol = self.torch_to_mol,
                    relaxed = relaxed,
                    ref_smiles = self.train_smiles,
                    test_smile = self.test_smiles
                ) if any([m in self.metrics_list for m in [
                    'molecular_validity',
                    'molecular_uniqueness',
                    'molecular_novelty',
                    "bond_distance_metric"
                ]]) else None,


                # apply post-hoc correction to molecules
                base_t.guarded_include(

                    if_ = any([m in self.metrics_list for m in ['fcd', 'nspdk']]),
                    do_ = base_t.DFCustomTransform(
                            free_transform=self.torch_to_mol,
                            src_df='data',
                            src_consts={
                                'override_relaxed': relaxed,
                                'override_post_hoc_mols_fix': True
                            },
                            dst_df='posthoc_molecules'
                    )
                ),

                # compute FCD if needed
                fcd_pipeline(
                    metrics = self.metrics,
                    mols_df = 'posthoc_molecules',
                    smiles_df = 'posthoc_smiles',
                    eval_smiles = self.eval_smiles
                ) if 'fcd' in self.metrics_list else None,

                # compute nspdk if needed
                nspdk_pipeline(
                    metrics = self.metrics,
                    mols_df = 'posthoc_molecules',
                    nx_df = 'posthoc_nx',
                    eval_nx = self.eval_nx
                ) if 'nspdk' in self.metrics_list else None
            ]
        )

    def call(self, data, **kwargs):
        return self.pipeline({'data': data, **kwargs})