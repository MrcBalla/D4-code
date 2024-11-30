from typing import Dict, List, Any, Callable
from src.test import reg_metrics

from torch import nn

class AssignmentException(Exception):
    pass


class Assignment(nn.Module):

    def __init__(self, metrics_list: Dict[str, Dict]):
        super().__init__()
        self._metrics = nn.ModuleDict()
        self._metrics_cfg = metrics_list
        self._metrics_list = []
        for metric in metrics_list:
            if isinstance(metric, dict):
                metric_name = list(metric.keys())[0]
                metric_fn = reg_metrics.get_instance_from_dict(metric)
                
            else:
                metric_name = metric
                metric_fn = reg_metrics.get_instance(metric)
                

            self._metrics[metric] = metric_fn
            self._metrics_list.append(metric_name)

        self._additional_data = {}


    @property
    def metrics(self) -> Dict[str, Callable]:
        return self._metrics
    
    @property
    def metrics_list(self) -> List[str]:
        return self._metrics_list
    
    @property
    def metrics_params(self) -> Dict[str, Dict]:
        return self._metrics_cfg
    
    @property
    def additional_data(self) -> Dict:
        return self._additional_data
    

    def add_datafield(self, name: str, value: Any):
        self._additional_data[name] = value

    
    def check(self, metrics: Dict[str, Any]):
        for m in self._metrics:
            if m not in metrics:
                raise AssignmentException(f'Assignment not complete: metric {m} not found in metrics')


    def call(self, data, **kwargs):
        raise NotImplementedError
        
    def __call__(self, **kwargs):
        out = self.call(**kwargs)
        self.check(out)
        return {m: out[m] for m in self._metrics_list}