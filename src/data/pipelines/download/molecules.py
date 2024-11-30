
from torch_geometric.data import extract_zip, extract_gz

import src.data.transforms as base_t
from src.data.transforms import (
    downloads as down_t
)
import src.data.dataset as ds

from src.data.pipelines import reg_download

@reg_download.register('qm9')
class QM9DownloadPipeline:

    def __init__(self):
        pass


    def __repr__(self):
        return 'QM9DownloadPipeline()'


    def __call__(self) -> base_t.DFPipeline:
        
        pipeline = base_t.DFPipeline(

            output_files = {
                'mols_file': 'gdb9.sdf',
                'targets_csv_file': 'gdb9.sdf.csv',
                'skip_csv_file': '3195404'
            },
            
            transforms=[
                #########################  FOLDERS SETUP  #########################
                base_t.DFAddDatafield('raw_path',       ds.STD_FOLDER_RAW),
                base_t.DFSPecializePath([ds.KEY_ROOT, 'raw_path'], 'raw_path'),
                
                base_t.DFCreateFolder(
                    destination_df =			'raw_path'
                ),

                ##################  DOWNLOAD AND UNZIP DATASETS  ##################
                down_t.DFDownloadFromURL(
                    download_to_df =	'raw_path',
                    url={
                        'qm9_data': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip',
                        'qm9_readme': 'https://ndownloader.figshare.com/files/3195404'
                    }
                ),
                down_t.DFExtract(
                    extr_method=extract_zip,
                    datafield='qm9_data',
                    extract_path_df='raw_path'
                )
            ]
        )

        return pipeline

@reg_download.register('gdb13')
class GDB13DownloadPipeline:

    def __init__(self):
        pass


    def __repr__(self):
        return 'GDB13DownloadPipeline()'


    def __call__(self) -> base_t.DFPipeline:
        
        pipeline = base_t.DFPipeline(

            output_files = {
                'mols_file': 'gdb13.rand1M.smi'
            },
            
            transforms=[
                #########################  FOLDERS SETUP  #########################
                base_t.DFAddDatafield('raw_path',       ds.STD_FOLDER_RAW),
                base_t.DFSPecializePath([ds.KEY_ROOT, 'raw_path'], 'raw_path'),
                
                base_t.DFCreateFolder(
                    destination_df =			'raw_path'
                ),

                ##################  DOWNLOAD AND UNZIP DATASETS  ##################
                down_t.DFDownloadFromURL(
                    download_to_df =	'raw_path',
                    url={
                        'gdb13_data': 'https://zenodo.org/record/5172018/files/gdb13.rand1M.smi.gz?download=1&ref=gdb.unibe.ch',
                    }
                ),
                down_t.DFExtract(
                    extr_method=extract_gz,
                    datafield='gdb13_data',
                    extract_path_df='raw_path'
                )
            ]
        )

        return pipeline
