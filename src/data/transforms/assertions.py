from typing import Dict, List, Any, Optional

from . import DFBaseTransform


class Assertion(DFBaseTransform):

    def __init__(self, message: str = None):

        add_message = '' if message is None else f': {message}'

        self.message = (
            f'Assertion failed for {self.__class__.__name__}'
            f'{add_message}'
        )
    

    def assertion(self, data: Dict) -> bool:
        raise NotImplementedError


    def __call__(self, data: Dict) -> Dict:
        assert self.assertion(data), self.message
        return data

    @property
    def output_df_list(self) -> List[str]:
        return ['assertion']
    


class AssertEqual(Assertion):

    def __init__(
            self,
            arg1_df: str,
            arg2_df: Optional[str] = None,
            value: Optional[Any] = None
        ):

        if arg2_df is None and value is None:
            raise ValueError('Either arg2_df or value must be provided')

        super().__init__(f'{arg1_df} == {arg2_df if arg2_df is not None else value}')
        self.arg1_df = arg1_df
        self.arg2_df = arg2_df
        self.value = value
        
        if arg2_df is None:
            self._input_df_list = [arg1_df]
        else:
            self._input_df_list = [arg1_df, arg2_df]

    def assertion(self, data: Dict) -> bool:
        if self.arg2_df is not None:
            return data[self.arg1_df] == data[self.arg2_df]
        else:
            return data[self.arg1_df] == self.value

    @property
    def input_df_list(self) -> List[str]:
        return self._input_df_list