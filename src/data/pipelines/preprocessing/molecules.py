from typing import List

import src.data.transforms as base_t
from src.data.transforms import (
    preprocessing as prep_t,
    splitting as split_t,
    conditions as cond_t,
    graphs as grph_t,
    molecular as chem_t,
    qm9 as qm9_t
)
from src.data.pipelines.preprocessing.common import graph_list_to_one_hot_transform
from src.data.pipelines.preprocessing.common import graph_list_padding_transform

import src.data.dataset as ds


from torch_geometric.datasets.qm9 import conversion
import numpy as np

from src.data.pipelines import reg_preprocess

def get_len(d: List) -> int:
    return len(d)
    
def get_shape_1(y) -> int:
    return y.shape[1]

@reg_preprocess.register('qm9')
class QM9PreprocessingPipeline:

    def __init__(self):
        pass


    def __repr__(self):
        return 'QM9PreprocessingPipeline()'


    def __call__(
            self,
            sanitize: bool=False,
            kekulize: bool=False,
            remove_hydrogens: bool=False,
            remove_nonsmiles: bool=True,
            aromatic_bond: bool=False,
            tr_val_test_split: bool=True,
            valid_split: int = 10000,
            test_split: int = 10000,
            compute_one_hot: bool = True
        ) -> base_t.DFPipeline:


        # SPLIT CASE
        # each output datafield is associated with its file
        # e.g. 'dataset_path_train' -> 'qm9_train_data.pt'
        # e.g. 'smiles_file_test' -> 'qm9_test_smiles.json'
        # NON-SPLIT CASE
        # each output datafield is associated with its file
        # e.g. 'dataset_path' -> 'qm9_data.pt'

        splits, out_files = split_t.get_outputdfs_files_splits(
            split_test = tr_val_test_split,
            split_valid = tr_val_test_split and valid_split is not None,
            outdf_file_ids_exts=[
                (ds.KEY_DATA_CONTAINER, 'data', 'pt'),
                (ds.KEY_DATA_INFO, 'info', 'json'),
                ('smiles_file', 'smiles', 'json'),
            ],
            file_format_string='qm9{split}_{file_id}.{ext}'
        )
        

        ############################################################################
        #                              PIPELINE START                              #
        ############################################################################
        pipeline = base_t.DFPipeline(

            input_files = {
                'mols_file': 'gdb9.sdf',
                'targets_csv_file': 'gdb9.sdf.csv',
                'skip_csv_file': '3195404'
            },

            output_files = out_files,
            
            transforms = [
            
                #########################  FOLDERS SETUP  #########################
                base_t.DFAddDatafield('raw_path',       ds.STD_FOLDER_RAW),
                base_t.DFAddDatafield('processed_path', ds.STD_FOLDER_PROCESSED),

                base_t.DFSPecializePath([ds.KEY_ROOT, 'raw_path'], 'raw_path'),
                base_t.DFSPecializePath([ds.KEY_ROOT, 'processed_path'], 'processed_path'),

                base_t.DFLocalizeFiles(
                    root_df = 'raw_path',
                    file_dfs = [
                        'mols_file',
                        'targets_csv_file',
                        'skip_csv_file'
                    ]
                ),
                base_t.DFLocalizeFiles(
                    root_df = 'processed_path',
                    file_dfs = list(out_files.keys())
                ),

                base_t.DFCreateFolder(
                    destination_df =    'processed_path'
                ),

                #####################  TARGETS PREPROCESSING  #####################
                prep_t.DFReadCSV(
                    datafield =         'targets_csv_file',
                    columns_df =        'targets_columns',
                    index_df =          'targets_index',
                    table_df =          'targets',
                    delimiter =         ','
                ),
                chem_t.DFApplyUnitConversion(
                    datafield =         'targets',
                    conversion_factors =conversion
                ),

                ######################  SETUP SKIP MOLECULES  ######################
                base_t.DFCustomTransform(
                    src_df =     'skip_csv_file',
                    dst_df =     'skip_list',
                    free_transform =    qm9_t.prepare_skip_list
                ),

                ####################  MOLECULES PREPROCESSING  ####################
                base_t.DFAddDatafield('mol_graphs', []),
                base_t.DFAddDatafield('smiles_list', []),
                chem_t.DFAddFeaturesSDF(
                    datafield =     'mols_file',
                    remove_hs=      remove_hydrogens
                ),
                chem_t.DFReadMolecules(
                    datafield =         'mols_file',
                    new_datafield =     'mols',
                    sanitize =          sanitize,           # pipeline input parameter
                    remove_hydrogens =  remove_hydrogens    # pipeline input parameter
                ),
                base_t.DFIterateOver(
                    datafield =         'mols',
                    iter_idx_df =       'curr_idx',
                    iter_elem_df =      'curr_mol',
                    use_tqdm =          True,
  
                    transform=base_t.DFCompose([
                            
                        ################  FILTER CORRECT MOLECULES  ################
                        base_t.DFConditional(
                            
                            #######################  FILTER  #######################
                            condition = cond_t.ManyConditions([
                                cond_t.CondNotNone(
                                    obj_to_check_df =       'curr_mol',
                                ),
                                cond_t.CondInList(
                                    obj_to_check_df =       'curr_idx',
                                    check_list_df =         'skip_list',
                                    check =                 'not in'
                                )
                            ]),
                            
                            #########  RDKIT MOLECULE TO TORCH GEOM GRAPH  #########
                            transform=base_t.DFCompose([

                                ############  APPLY PREPROC TO MOL (KEKULIZE)  ############
                                chem_t.DFMoleculePreprocess(
                                    molecule_df =           'curr_mol',
                                    kekulize =              kekulize
                                ),

                                chem_t.DFMolToSmiles(
                                    mol_df =                'curr_mol',
                                    smiles_df =             'full_smiles',
                                    sanitize_smiles =       False,
                                    isomeric=               False,
                                    canonical=              True
                                ),

                                ###############  MOL TO TORCH GRAPH  ###############
                                chem_t.DFAtomsData(
                                    molecule_df =           'curr_mol',
                                    result_nodes_df =       'curr_nodes',
                                    atom_types_df =         'atom_types'                                ),

                                chem_t.DFBondsData(
                                    molecule_df =           'curr_mol',
                                    result_edge_index_df =  'curr_edge_index',
                                    result_edge_attr_df =   'curr_edge_attr',
                                    bond_types_df =         'bond_types',
                                    aromatic_bond=           aromatic_bond
                                ),

                                base_t.DFIndexElement(
                                    datalist_df =           'targets',
                                    index_df =              'curr_idx',
                                    result_df =             'curr_targets'
                                ),

                                chem_t.DFAdditionaFeatureEdges(
                                    molecule_df = 'curr_mol',
                                    attribute_df_edge = 'attribute_edge'
                                ),
                                
                                chem_t.DFAdditionaCoordinate(
                                    molecule_df = 'curr_mol',
                                    attribute_df_pos = 'pos'   
                                ),

                                chem_t.DFAdditionaFeatureNodes(
                                    molecule_df= 'curr_mol',
                                    attribute_df_node= 'attribute_node',
                                    which=['Formal_charge']
                                ),   

                                chem_t.DFToGraph(
                                    'attribute_edge', "attribute_node", "pos",
                                    result_df =             'res_graph',
                                    nodes_df =              'curr_nodes',
                                    edge_index_df =         'curr_edge_index',
                                    edge_attr_df =          'curr_edge_attr',
                                    targets_df =            'curr_targets',
                                    atom_types =            'atom_types',
                                    remove_hydrogen=        remove_hydrogens
                                ),
                                #################  GRAPH TO SMILES  ################
                                chem_t.DFGraphToMol(
                                    graph_df =              'res_graph',
                                    result_mol_df =         'res_mol',
                                    atom_decoder_df =       'atom_types',
                                    bond_decoder_df =       'bond_types'
                                ),
                                chem_t.DFMolToSmiles(
                                    mol_df =                'res_mol',
                                    smiles_df =             'res_smiles',
                                    sanitize_smiles =       False,
                                    isomeric=               False,
                                    canonical=              True
                                ),

                                ################  FILTER INVALIDS  #################
                                # the filter is added only if remove_nonsmiles is True
                                # otherwise, the conditional is always True
                                base_t.DFConditional(
                                    condition = base_t.guarded_include(
                                        if_ = remove_nonsmiles, # pipeline input parameter
                                        do_ = chem_t.CondValidSMILES('res_smiles')
                                    ),
                                    transform = base_t.DFCompose([
                                        # compute molecular properties on the actual molecule
                                        # computed from the graph
                                    
                                        base_t.DFAppendElement(
                                            elem_df =               'full_smiles', # use correct smiles
                                            datalist_df =           'smiles_list'
                                        ),
                                        base_t.DFAppendElement(
                                            elem_df =                'res_graph',
                                            datalist_df =            'mol_graphs'
                                        )
                                    ])
                                )

                            ])
                        )
                    ])
                ),
                chem_t.AssertAtomTypes(
                    atom_types = 'atom_types',
                    remove_hydrogen= remove_hydrogens
                ),
                # collect number of atom types
                base_t.DFCustomTransform(
                    src_df =            'atom_types',
                    dst_df =            'num_cls_nodes',
                    free_transform =    get_len
                ),
                # collect number of bond types
                base_t.DFCustomTransform(
                    src_df =            'bond_types',
                    dst_df =            'num_cls_edges',
                    free_transform =    get_len
                ),
                
                # collect number of target components
                base_t.DFCustomTransform(
                    src_df =            'targets',
                    dst_df =            'dim_targets',
                    free_transform =    get_shape_1
                ),

                # insert in datafield the maximum number of nodes, to perform 
                # padding
                grph_t.DFGGetMaxStatistics(
                    list_of_graphs_df = 'mol_graphs',
                    max_number_nodes = 'max_number_of_nodes'
                ),

                # transform all graphs features to one-hot
                # if enabled (compute_onehot = True)
                graph_list_to_one_hot_transform(
                    graph_list_df =         'mol_graphs',
                    num_classes_node_df =   'num_cls_nodes',
                    num_classes_edge_df =   'num_cls_edges',
                    enable =                compute_one_hot
                ),

                graph_list_padding_transform(
                    graph_list_df =         'mol_graphs',
                    max_nodes=               'max_number_of_nodes'
                ),

                ###########  IF SHOULD SPLIT, SPLIT INTO DATASET SPLITS  ###########
                base_t.guarded_include(
                    if_ = tr_val_test_split, # pipeline input parameter
                    do_ = split_t.DFSplitTrainTestValid(
                        data_list_df =        ['mol_graphs',   'smiles_list'],
                        test_fraction =        test_split,     # pipeline input parameter
                        valid_fraction =    valid_split,    # pipeline input parameter
                        test_df =            ['graphs_test',  'smiles_test'],
                        valid_df =            ['graphs_valid', 'smiles_valid'],
                        train_df =            ['graphs_train', 'smiles_train']
                    )
                ),

                #######################  SPLITTING PIPELINE  #######################
                split_t.DFForEachSplit(
                    split_names =   splits,
                    dont_split = not tr_val_test_split,

                    transform =     base_t.DFCompose([
                        ###################  SAVE GRAPHS DATASET  ##################
                        grph_t.DFSaveGraphListTorch(
                            graph_list_df =     'graphs{split}',
                            save_path_df =      ds.KEY_DATA_CONTAINER + '{split}'
                        ),

                        ###################  SAVE DATASET INFOS  ###################
                        # collect min and max number of nodes
                        grph_t.DFCollectGraphNodesStatistics(
                            list_of_graphs_df = 'graphs{split}',
                            df_stats_dict = {
                                'num_nodes_min': np.min,
                                'num_nodes_max': np.max
                            },
                            histogram_df =      'num_nodes_hist'
                        ),
                        # collect number of molecules
                        base_t.DFCustomTransform(
                            src_df =     'graphs{split}',
                            dst_df =     'num_molecules',
                            free_transform =     get_len
                        ),
                        # collect number of SMILES
                        base_t.DFCustomTransform(
                            src_df =     'smiles{split}',
                            dst_df =     'num_smiles',
                            free_transform =     get_len
                        ),
                        # save the above statistics to file as
                        # a dictionary
                        base_t.DFSaveToFile(
                            save_path_df =      ds.KEY_DATA_INFO + '{split}',
                            datafields = [
                                'atom_types',
                                'bond_types',
                                'num_nodes_min',
                                'num_nodes_max',
                                'num_nodes_hist',
                                'num_cls_nodes',
                                'num_cls_edges',
                                'dim_targets',
                                'num_molecules',
                                'num_smiles'
                            ]
                        ),
                        base_t.DFSaveToFile(
                            save_path_df =      'smiles_file{split}',
                            datafields =         ['smiles{split}']
                        )
                    ]),
                )
            ]
        )

        return pipeline
    

@reg_preprocess.register('gdb13')
class GDB13PreprocessingPipeline:

    def __init__(self):
        pass


    def __repr__(self):
        return f'GDB13PreprocessingPipeline({self.which_zinc})'


    def __call__(
            self,
            sanitize: bool=False,
            kekulize: bool=False,
            remove_hydrogens: bool=False,
            remove_nonsmiles: bool=True,
            aromatic_bond: bool=False,
            tr_val_test_split: bool=True,
            valid_split: int = 10000,
            test_split: int = 10000,
            compute_one_hot: bool = True
        ) -> base_t.DFPipeline:


        splits, out_files = split_t.get_outputdfs_files_splits(
            split_test = tr_val_test_split,
            split_valid = tr_val_test_split and valid_split is not None,
            outdf_file_ids_exts=[
                (ds.KEY_DATA_CONTAINER, 'data', 'pt'),
                (ds.KEY_DATA_INFO, 'info', 'json'),
                ('smiles_file', 'smiles', 'json'),
            ],
            file_format_string='zinc{split}_{file_id}.{ext}'
        )
        

        ############################################################################
        #                              PIPELINE START                              #
        ############################################################################
        pipeline = base_t.DFPipeline(

            input_files = {
                'mols_file': "gdb13.rand1M.smi",
            },

            output_files = out_files,
            
            transforms = [
            
                #########################  FOLDERS SETUP  #########################
                base_t.DFAddDatafield('raw_path',       ds.STD_FOLDER_RAW),
                base_t.DFAddDatafield('processed_path', ds.STD_FOLDER_PROCESSED),

                base_t.DFSPecializePath([ds.KEY_ROOT, 'raw_path'], 'raw_path'),
                base_t.DFSPecializePath([ds.KEY_ROOT, 'processed_path'], 'processed_path'),

                base_t.DFLocalizeFiles(
                    root_df = 'raw_path',
                    file_dfs = [
                        'mols_file'
                    ]
                ),
                base_t.DFLocalizeFiles(
                    root_df = 'processed_path',
                    file_dfs = list(out_files.keys())
                ),

                base_t.DFCreateFolder(
                    destination_df =    'processed_path'
                ),

                #####################  TARGETS PREPROCESSING  #####################
                chem_t.DFReadMolecules(
                    datafield =         'mols_file',
                    new_datafield =     'mols',
                    sanitize =          sanitize,           # pipeline input parameter
                    remove_hydrogens =  remove_hydrogens    # pipeline input parameter
                ),

                ####################  MOLECULES PREPROCESSING  ####################
                base_t.DFAddDatafield('mol_graphs', []),
                base_t.DFAddDatafield('smiles_list', []),
                base_t.DFIterateOver(
                    datafield =         'mols',
                    iter_idx_df =       'curr_idx',
                    iter_elem_df =      'curr_mol',
                    use_tqdm =          True,

                    transform=base_t.DFCompose([
            
                        ############  APPLY PREPROC TO MOL (KEKULIZE)  ############
                        chem_t.DFMoleculePreprocess(
                                    molecule_df =           'curr_mol',
                                    kekulize =              True
                                ),

                        chem_t.DFMolToSmiles(
                                    mol_df =                'curr_mol',
                                    smiles_df =             'full_smiles',
                                    sanitize_smiles =       False,
                                    isomeric=               False,
                                    canonical=              True
                                ),

                        ###############  MOL TO TORCH GRAPH  ###############
                        chem_t.DFAtomsData(
                            molecule_df =           'curr_mol',
                            result_nodes_df =       'curr_nodes',
                            atom_types_df =         'atom_types'
                        ),
                        chem_t.DFBondsData(
                            molecule_df =           'curr_mol',
                            result_edge_index_df =  'curr_edge_index',
                            result_edge_attr_df =   'curr_edge_attr',
                            bond_types_df =         'bond_types',
                            aromatic_bond= False
                        ),
                          
                        chem_t.DFAdditionaCoordinate_zinc(
                                    molecule_df = 'curr_mol',
                                    attribute_df_pos = 'pos'   
                                ),
                        
                        chem_t.DFAdditionaFeatureEdges_zinc(
                                    molecule_df = 'curr_mol',
                                    attribute_df_edge = 'attribute_edge'
                                ),

                        chem_t.DFAdditionaFeatureNodes_sdf(
                                    molecule_df= 'curr_mol',
                                    attribute_df_node= 'attribute_node',
                                    which=['Formal_charge']
                                ),   
                        
                        chem_t.DFToGraph(
                                    'attribute_edge', "attribute_node", "pos",
                                    result_df =             'res_graph',
                                    nodes_df =              'curr_nodes',
                                    edge_index_df =         'curr_edge_index',
                                    edge_attr_df =          'curr_edge_attr',
                                    targets_df =            'curr_targets',
                                    atom_types =            'atom_types',
                                    remove_hydrogen=        remove_hydrogens
                                ),

                        # append smiles and graph to data lists
                        base_t.DFAppendElement(
                            elem_df =                'full_smiles',
                            datalist_df =            'smiles_list'
                        ),
                        base_t.DFAppendElement(
                            elem_df =                'res_graph',
                            datalist_df =            'mol_graphs'
                        )
                    ])
                ),

                # collect number of atom types
                base_t.DFCustomTransform(
                    src_df =     'atom_types',
                    dst_df =     'num_cls_nodes',
                    free_transform =     get_len
                ),
                # collect number of bond types
                base_t.DFCustomTransform(
                    src_df =     'bond_types',
                    dst_df =     'num_cls_edges',
                    free_transform =     get_len
                ),
                grph_t.DFGGetMaxStatistics(
                    list_of_graphs_df = 'mol_graphs',
                    max_number_nodes = 'max_number_of_nodes'
                ),

                # transform all graphs features to one-hot
                # if enabled (compute_onehot = True)
                graph_list_to_one_hot_transform(
                    graph_list_df =         'mol_graphs',
                    num_classes_node_df =   'num_cls_nodes',
                    num_classes_edge_df =   'num_cls_edges',
                    enable =                compute_one_hot
                ),
                graph_list_padding_transform(
                    graph_list_df =         'mol_graphs',
                    max_nodes=               'max_number_of_nodes'
                ),

                ###########  IF SHOULD SPLIT, SPLIT INTO DATASET SPLITS  ###########
                base_t.guarded_include(
                    if_ = tr_val_test_split, # pipeline input parameter
                    do_ = split_t.DFSplitTrainTestValid(
                        data_list_df =        ['mol_graphs',   'smiles_list'],
                        test_fraction =        test_split,     # pipeline input parameter
                        valid_fraction =    valid_split,    # pipeline input parameter
                        test_df =            ['graphs_test',  'smiles_test'],
                        valid_df =            ['graphs_valid', 'smiles_valid'],
                        train_df =            ['graphs_train', 'smiles_train']
                    )
                ),

                #######################  SPLITTING PIPELINE  #######################
                split_t.DFForEachSplit(
                    split_names =   splits,
                    dont_split = not tr_val_test_split,

                    transform =     base_t.DFCompose([
                        ###################  SAVE GRAPHS DATASET  ##################
                        grph_t.DFSaveGraphListTorch(
                            graph_list_df =     'graphs{split}',
                            save_path_df =      ds.KEY_DATA_CONTAINER + '{split}'
                        ),

                        ###################  SAVE DATASET INFOS  ###################
                        # collect min and max number of nodes
                        grph_t.DFCollectGraphNodesStatistics(
                            list_of_graphs_df = 'graphs{split}',
                            df_stats_dict = {
                                'num_nodes_min': np.min,
                                'num_nodes_max': np.max
                            },
                            histogram_df =      'num_nodes_hist'
                        ),
                        # collect number of molecules
                        base_t.DFCustomTransform(
                            src_df =     'graphs{split}',
                            dst_df =     'num_molecules',
                            free_transform =     get_len
                        ),
                        # collect number of SMILES
                        base_t.DFCustomTransform(
                            src_df =     'smiles{split}',
                            dst_df =     'num_smiles',
                            free_transform =     get_len
                        ),
                        # save the above statistics to file as
                        # a dictionary
                        base_t.DFSaveToFile(
                            save_path_df =      ds.KEY_DATA_INFO + '{split}',
                            datafields = [
                                'atom_types',
                                'bond_types',
                                'num_nodes_min',
                                'num_nodes_max',
                                'num_nodes_hist',
                                'num_cls_nodes',
                                'num_cls_edges',
                                'num_molecules',
                                'num_smiles'
                            ]
                        ),
                        base_t.DFSaveToFile(
                            save_path_df =      'smiles_file{split}',
                            datafields =        ['smiles{split}']
                        )
                    ]),
                )
            ]
        )

        return pipeline
  