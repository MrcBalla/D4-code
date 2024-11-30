from typing import List, Dict, Optional
from abc import ABC, abstractmethod

import pytorch_lightning as pl

from src.test.assignment import Assignment
from logging import Logger


class Generator(ABC, pl.LightningModule):

    IGNORED_HPARAMS = [
        'dataset_info',
        'test_assignment',
        'console_logger'
    ]

    def __init__(
            self,
            dataset_info: Dict = None,
            test_assignment: Assignment = None,
            console_logger: Logger = None
        ):
        super().__init__()

        self.dataset_info = dataset_info
        self.test_assignment = test_assignment
        self.console_logger = console_logger
        


    @abstractmethod
    def sample(
        self,
        num_samples: int,
        condition: Optional[Dict]=None,
        **kwargs
    ):
        raise NotImplementedError