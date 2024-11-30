from typing import Dict

import json


from src.data.datamodule import GraphDataModule
import src.data.transforms as base_t
from src.data.transforms import (
    graphs as grph_t,
    assertions as assert_t
)
from src.data.simple_transforms.graph import graph2nx


from src.test import reg_assignment
from src.test.assignment import Assignment
from src.test.metrics.sampling import contains_sampling_metrics, is_sampling_metric
from src.test.utils.synth import dataloader_to_nx



@reg_assignment.register('graph')
class GenericGraphAssignment(Assignment):

    def __init__(
            self,
            metrics_list: Dict[str, Dict],
            split: str,
            datamodule: GraphDataModule,
            how_many_to_generate: int
        ):
        super().__init__(metrics_list)

        self.how_many_to_generate = how_many_to_generate

        has_sampling = contains_sampling_metrics(self.metrics)

        test_graphs = []
        if has_sampling:
            test_graphs = dataloader_to_nx(datamodule.get_dataloader(split))

        self.pipeline = base_t.DFPipeline(
            
            input_dfs = ['data'],
            output_dfs = self.metrics_list,

            transforms=[

                # check for size if metrics contains sampling metrics
                base_t.guarded_include(
                    if_ = has_sampling,
                    do_ = base_t.DFCompose([
                        # base_t.DFCustomTransform(len, 'data', 'data_len'),
                        # assert_t.AssertEqual(
                        #     'data_len', value=how_many_to_generate
                        # ),

                        # Convert graphs to nx
                        grph_t.DFGraphToNetworkx(
                            list_of_graphs_df = 'data',
                            list_of_nxgraphs_df = 'nx_graphs'
                        ),
                        
                        base_t.DFCompose([
                            base_t.DFCustomTransform(
                                free_transform=metric_fn.override(
                                    test_graphs = test_graphs
                                ),
                                src_df='nx_graphs',
                                dst_df={metric_name: metric_name}
                            )
                            for metric_name, metric_fn in self.metrics.items()
                            if is_sampling_metric(metric_fn)
                        ])
                    ])
                )

            ]

        )
        

    def call(self, data, **kwargs):
        return self.pipeline({'data': data, **kwargs})