from torch_geometric.data import extract_zip

import src.data.transforms as base_t
from src.data.transforms import (
    downloads as down_t
)
import src.data.dataset as ds

from src.data.pipelines import registered_download_pipeline

class Qm9DownloadPipeline:



    def __call__(
            self
        ) -> base_t.DFPipeline:
    
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
    

@registered_download_pipeline
class Qm9DownloadPipeline:


    def __call__(
            self
        ) -> base_t.DFPipeline:

        # the following is pseudo-code
        pipeline = base_t.DFPipeline(

            input_dfs = ['ds_root'],
            outputs = ['mols_file', 'targets_csv_file', 'skip_csv_file'],

            transforms = [

                # create the raw folder
                base_t.DFAppendPath('raw').inp('ds_root').out('raw_root'),
                base_t.DFCreateFolder().inp('raw_root'),


                # download the data
                down_t.DFDownloadFromURL(
                    urls = [
                        'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip',
                        'https://ndownloader.figshare.com/files/3195404'
                    ]
                )\
                    .inp('raw_path')\
                    .out('qm9_data', 'qm9_readme'),


                # extract the data
                down_t.DFExtract(extract_zip).inp('qm9_data').out('raw_path'),

                # set the output files
                base_t.DefVar('gdb9.sdf').out('mols_file'),
                base_t.DefVar('gdb9.sdf.csv').out('targets_csv_file'),
                base_t.DefVar('3195404').out('skip_csv_file')

            ]
        )

        return pipeline

        