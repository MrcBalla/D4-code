from typing import Union, List, Dict, Callable, Optional

import os
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional
import gdown

import fsspec

from torch_geometric.data import download_url

from . import DFBaseTransform


DF_RAW_FOLDER = 'raw_folder'
DF_RAW_FILE_NAMES = 'raw_file_names'

class DownloadPipelineException(Exception):
    pass


class DFDownloadFromURL(DFBaseTransform):
            
    def __init__(
            self,
            url: Union[Dict[str, str], List[str], str],
            download_to_df: Optional[str]=None,
        ):
        self.download_to_df = download_to_df

        if isinstance(url, str):
            urls = {'download_path': url}
        elif isinstance(url, list):
            urls = {f'download_path_{i}': url for i, url in enumerate(url)}
        elif isinstance(url, dict):
            urls = url
        else:
            raise DownloadPipelineException(f'Url in {self.__class__.__name__} object has an invalid type. Should be: str, list, dict. Found: {type(url)}')
        
        self.urls = urls

    
    def __call__(self, data: Dict) -> Dict:

        raw_folder = data[self.download_to_df]

        for name, url in self.urls.items():
            file_path = download_url(url, raw_folder)
            data[name] = file_path

        return data
    
    @property
    def output_df_list(self) -> List[str]:
        return [self.download_to_df]

    def args_repr(self) -> str:
        urls_string = '\n'.join(list(self.urls.values()))
        return (
            f'urls=[\n{urls_string}\n]'
        )
    
class DFDownloadFromGoogleDrive(DFBaseTransform):
            
    def __init__(
            self,
            url: Union[Dict[str, str], List[str], str],
            download_to_df: Optional[str]=None,
        ):
        self.download_to_df = download_to_df

        if isinstance(url, str):
            urls = {'download_path': url}
        elif isinstance(url, list):
            urls = {f'download_path_{i}': url for i, url in enumerate(url)}
        elif isinstance(url, dict):
            urls = url
        else:
            raise DownloadPipelineException(f'Url in {self.__class__.__name__} object has an invalid type. Should be: str, list, dict. Found: {type(url)}')
        
        self.urls = urls


    def download_google_url(
        url: str,
        folder: str,
        log: bool = True,
        filename: Optional[str] = None,
    ):
        r"""Downloads the content of an google URL to a specific folder.

        Args:
            url (str): The URL.
            folder (str): The folder.
            log (bool, optional): If :obj:`False`, will not print anything to the
                console. (default: :obj:`True`)
            filename (str, optional): The filename of the downloaded file. If set
                to :obj:`None`, will correspond to the filename given by the URL.
                (default: :obj:`None`)
        """
        if filename is None:
            filename = url.rpartition('/')[2]
            filename = filename if filename[0] == '?' else filename.split('?')[0]

        path = osp.join(folder, filename)

        if os.path.exists(path):  # pragma: no cover
            if log and 'pytest' not in sys.modules:
                print(f'Using existing file {filename}', file=sys.stderr)
            return path

        if log and 'pytest' not in sys.modules:
            print(f'Downloading {url}', file=sys.stderr)

        os.makedirs(folder, exist_ok=True)

        # downloading from Google
        gdown.download(url, path)

        return path

    
    def __call__(self, data: Dict) -> Dict:

        raw_folder = data[self.download_to_df]

        for name, url in self.urls.items():
            file_path = DFDownloadFromGoogleDrive.download_google_url(url, raw_folder, filename="dataset.zip")
            data[name] = file_path
        return data
    
    @property
    def output_df_list(self) -> List[str]:
        return [self.download_to_df]

    def args_repr(self) -> str:
        urls_string = '\n'.join(list(self.urls.values()))
        return (
            f'urls=[\n{urls_string}\n]'
        )
    
    
KEY_EXTRACTED = '_extracted'

class DFExtract(DFBaseTransform):

    def __init__(
            self,
            extr_method: Callable[[str, str], None],
            datafield: Union[List[str], str],
            extract_path_df: Optional[str]=None,
        ):
        self.extract_path_df = extract_path_df
        self.extr_method = extr_method

        if isinstance(datafield, str):
            datafields = [datafield]
        elif isinstance(datafield, List):
            datafields = datafield
        else:
            raise DownloadPipelineException(f'datafield in {self.__class__.__name__} object has an invalid type. Should be: str, list. Found: {type(datafield)}')
        
        self.datafields = datafields

    def _curr_files(self, folder: str) -> List[str]:
        return os.listdir(folder)

    def _new_files(self, folder: str, prev_files: List[str]) -> Union[List[str], List[str]]:
        curr_files = self._curr_files(folder)
        new_files = [
            f for f in curr_files
            if f not in prev_files
        ]
        return new_files, curr_files

    def __call__(self, data: Dict) -> Dict:

        raw_folder = data[self.extract_path_df]

        prev_files = self._curr_files(raw_folder)

        for df in self.datafields:
            try:
                # get compressed file paths in data
                compr_file_path = data[df]
                # extract the file using the selected method
                self.extr_method(compr_file_path, raw_folder)

                # find new files and current files
                # to be used later
                new_files, prev_files = self._new_files(raw_folder, prev_files)

                # set the extracted files in data
                data[df + KEY_EXTRACTED] = new_files
            except Exception as ex:
                print('Something went wrong during the extraction of '\
                    +f'"{df}" at path "{data[df]}". Skipping '\
                    +f'this file. The thrown exception is: "{ex}"')
                
        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.extract_path_df] + self.datafields
    @property
    def output_df_list(self) -> List[str]:
        return [df + KEY_EXTRACTED for df in self.datafields]

    def args_repr(self) -> str:
        urls_string = ', '.join(self.datafields)
        return (
            f'urls=[\n{urls_string}\n]'
        )
