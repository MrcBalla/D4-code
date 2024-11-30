# distance __init__ file


##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/8757353a61235fa499dea0cbcd4771eb79b22901/dgd/diffusion_model_discrete.py
#
##########################################################################################################

from typing import Dict, Tuple, Union, Optional, List, Callable, Any

import time
import os
from copy import deepcopy

from logging import Logger
import wandb

import numpy as np

# utils for debugging
import sys

################  TORCH IMPORTS  #################
import torch
from torch import Tensor, BoolTensor, IntTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from torchmetrics import Metric

##############  DATATYPES IMPORTS  ###############
from src.datatypes import (
    dense,
    sparse,
    split
)
from src.datatypes.dense import DenseGraph, DenseEdges
from src.datatypes.sparse import SparseGraph, SparseEdges

################  NOISE IMPORTS  #################
from src.noise.timesample import (
    resolve_timesampler
)

from src.noise.graph_diffusion_distance import (
    resolve_graph_diffusion_process,
    resolve_graph_diffusion_schedule
)
from src.noise.config_support import build_noise_process
from src.noise.batch_transform.sequence_sampler import sample_sequences

###############  METRICS IMPORTS  ################
import src.test.metrics as m_list
from src.models.distance_gdb13.losses.train_denoising import TrainLoss_distance


from src.models.generation import Generator
from src.models import reg_models, reg_architectures
from src.test.assignment import Assignment
from src.datatypes.features import get_features_list
from src.datatypes.features.core import increase_dims_list, increase_dims, Feature

from src.models.architectures.gnn.distance_graph_transformer_gdb13 import distance_GraphTransformer_gdb13
from src.noise.graph_diffusion_distance import DiscreteUniformDiffusionProcess_distance

from src.models.distance_gdb13 import labels

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.aggregation import MeanMetric
from torchmetrics.regression import MeanAbsoluteError

from src.models.architectures.distributions.empirical import EmpiricalSampler
from pytorch_lightning.loggers import WandbLogger
import collections


KEY_TRAIN = 'TRAIN'
KEY_VALID = 'VALID'
KEY_TEST = 'TEST'

@reg_models.register()
class DistanceDenoisingDiffusionModel_gdb13(Generator):

    def __init__(
            self,

            ########### configurations ###########
            # model configurations
            denoising: Dict,
            diffusion: Dict,

            # optimizer configuration
            optimizer: Dict,
            
            # features configurations
            features: Dict = None,

            # generation configuration
            # e.g., conditional, batch size
            generation: Dict = None,

            # validation config
            validation: Dict = None,
            
            discard_conditioning: bool = True,
            received_dims: Optional[Dict] = None,

            ######## passed by configurator ######
            dataset_info: Dict = None,
            test_assignment: Assignment = None,
            console_logger: Logger = None
        ):
        
        super().__init__(
            dataset_info=dataset_info,
            test_assignment=test_assignment,
            console_logger=console_logger
        )

        ############################  CONFIGS SETUP  ###########################
        
        # setup console logger
        self.console_logger = console_logger

        # setup config on how to build the model and noise processes
        self.denoising_config = denoising
        self.diffusion_config = diffusion

        # setup optimizer configuration
        self.optimizer_config = optimizer

        # setup additional features
        self.additional_features: List[Feature] = get_features_list(features) if features else []

        # setup generation
        self.generation_config = generation

        self.validation_config = validation

        #######################  GRAPHS DIMENSIONS SETUP  ######################
        # setup model input and output dimensions (based on the dataset)
        self.data_dims = {
            'x': dataset_info['num_cls_nodes'],
            'e': dataset_info['num_cls_edges'],
            'y': 0 if discard_conditioning else dataset_info['dim_targets'],
            "c": 3,
            "dist": 1
        }

        self.data_dims['e'] += 1  # account for no-edge class

        if received_dims:
            self.received_dims = deepcopy(received_dims)
            self.received_dims['e'] += 1
        else:
            self.received_dims = self.data_dims

        # increase dimensions based on additional features (creates a copy)
        self.augmented_dims = increase_dims_list(self.received_dims, self.additional_features)
        
        self.augmented_dims = increase_dims(self.augmented_dims, {
            'y': 1  # account for diffusion time as a global y feature
        })

        self.console_logger.info(f'{self.__class__.__name__} dimensions:')
        self.console_logger.info(f"Size of input features: {self.augmented_dims}")
        self.console_logger.info(f"Size of output features: {self.data_dims}")


        ########################  BUILD DENOISING MODEL  #######################
        # use an empirical sampler when the number of nodes is not known
        self.empirical_sampler = EmpiricalSampler(
            dataset_info =      dataset_info,
            device =            self.device
        )

        # by default, the architecture is a GraphTransformer
        self.denoising_model = reg_architectures.get_instance_from_dict(
            config =        self.denoising_config.architecture,
            input_dims =    self.augmented_dims,
            output_dims =   self.data_dims,
        )

        ######################  BUILD DIFFUSION PROCESS  #######################
        self.diffusion_process: DiscreteUniformDiffusionProcess_distance
        self.diffusion_process, self.diffusion_timesampler = build_noise_process(
            config =                self.diffusion_config,
            process_resolver =      resolve_graph_diffusion_process,
            schedule_resolver =     resolve_graph_diffusion_schedule,
            timesampler_resolver =  resolve_timesampler
        )

        ######################  BUILD LOSSES AND METRICS  ######################
        self.train_loss = TrainLoss_distance(
            **self.denoising_config.loss
        )

        metrics = nn.ModuleDict({
            labels.DENOISE_CE_X: MeanMetric(),
            labels.DENOISE_CE_E: MeanMetric(),
            labels.DENOISE_CE_EXT_E: MeanMetric(),
            labels.DENOISE_ACC_X: MulticlassAccuracy(num_classes=self.data_dims['x'], validate_args=False),
            labels.DENOISE_ACC_E: MulticlassAccuracy(num_classes=self.data_dims['e'], validate_args=False),
            labels.DENOISE_ACC_EXT_E: MulticlassAccuracy(num_classes=self.data_dims['e'], validate_args=False),
            labels.DENOISE_CE_C: MeanMetric(),
            labels.DENOISE_ACC_C: MulticlassAccuracy(num_classes=3, validate_args=False),
            labels.DENOISE_MSE_DIST: MeanMetric(),
            labels.DENOISE_MAE_DIST: MeanAbsoluteError(),
            labels.DENOISE_TOTAL: MeanMetric()
        })

        self.metrics = nn.ModuleDict({
            KEY_TRAIN: deepcopy(metrics),
            KEY_VALID: deepcopy(metrics),
            KEY_TEST: deepcopy(metrics)
        })

        ############################  EXTRA SETUP  #############################

        self.start_time = None
        self.total_elapsed_time = 0
        self.max_memory_reserved = 0

        # save hyperaparameters (but those not in the Generator ignored list)
        self.save_hyperparameters(ignore=Generator.IGNORED_HPARAMS + ['received_dims'])


    def is_conditional(self):
        return self.generation_config['conditional']
    
    def get_external_nodes_dim(self):
        return self.denoising_model.get_external_nodes_dim()


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        overtime = 0 if self.start_time is None else time.time() - self.start_time
        checkpoint['total_elapsed_time'] = self.total_elapsed_time + overtime
        checkpoint['max_memory_reserved'] = max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.total_elapsed_time = checkpoint['total_elapsed_time']
        self.max_memory_reserved = checkpoint['max_memory_reserved']


    ############################################################################
    #                 SHORTHANDS FOR TRAINING/VALIDATION STEPS                 #
    ############################################################################
    
    
    
    def compute_true_pred_denoising(
            self,
            batch_to_generate: SparseGraph,
            batch_external: Optional[SparseGraph] = None,
            edges_external: Optional[SparseEdges] = None,
            train_step=False
        ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate the true and predicted nodes and egdes for the denoising
        process. The flow is as follows:
        1 - encode the batch_external to get encoded nodes
        2 - densify batch_to_generate as a DenseGraph, the encoded nodes,
            and the external edges, with onehot and masking
        3 - sample the diffusion process at uniformly random timesteps to
            make a noisy version of batch_to_generate (again requires onehot
            and masking)
        4 - try to denoise the above data which include the batch_to_generate
            and edges_external
        5 - flatten and pack the true and predicted nodes and edges
        The final order is: nodes, edges, external_edges.
        Predicted values are in expanded form, true values are collapsed. This is
        ideal for the cross-entropy loss function.

        Parameters
        ----------
        batch_to_generate : SparseGraph
            sparse graph with collapsed classes (i.e. class indices). This graph
            will be noised and denoised.
        batch_external : Optional[SparseGraph]
            sparse graph with onehot classes. The nodes of this graph will be
            encoded and used to denoise the batch_to_generate. Default is None,
            in which case only the batch_to_generate is noised and denoised.
        edges_external : Optional[Tuple[Tensor, Tensor]]
            external edges in edge_index and edge_attr form, to be noised and
            denoised. Default is None, in which case only the batch_to_generate
            is noised and denoised.

        Returns
        -------
        true_values : List[Tensor]
            list of true values of nodes and edges, in collapsed form.
        pred_values : List[Tensor]
            list of predicted values of nodes and edges, in expanded form.
        """

        ####################  FORMAT INPUT FOR PREDICTION  #####################
        # 1 - densify
        # transform the current nodes to dense format
        # transform the external nodes and edges to dense format if needed
        batch_to_generate_dense: DenseGraph
        ext_x: Tensor               # None if no external graph
        ext_node_mask: BoolTensor   # None if no external graph
        ext_adjmat: DenseEdges      # None if no external graph
        batch_to_generate_dense, ext_x, ext_node_mask, ext_adjmat, _ = format_generation_task_data(
            curr_graph =		batch_to_generate,
            ext_graph =         batch_external,
            edges_curr_ext =	edges_external
        )
        
        # setup masks for edges
        node_mask = batch_to_generate_dense.node_mask
        triang_edge_mask = torch.tril(batch_to_generate_dense.edge_mask, diagonal=-1)

        # 2 - copy true masked data (to be returned later)
        true_x = batch_to_generate_dense.x.argmax(dim=-1)[node_mask]
        true_e = batch_to_generate_dense.edge_adjmat.argmax(dim=-1)[triang_edge_mask]
        true_dist = batch_to_generate_dense.attribute_edge[triang_edge_mask]
        true_c = batch_to_generate_dense.attribute_node.argmax(dim=-1)[node_mask]

        if ext_adjmat is not None:
            ext_edge_mask = ext_adjmat.edge_mask
            true_ext_e = ext_adjmat.edge_adjmat.argmax(dim=-1)[ext_edge_mask]
        else:
            ext_edge_mask = None
            true_ext_e = None

        #######################  APPLY GRAPH DIFFUSION  ########################
        # sample the timesteps for the diffusion process
        max_times = torch.full((batch_to_generate.num_graphs,), self.diffusion_process.get_max_time()-1) # must be in cpu
        u: Tensor = self.diffusion_timesampler.sample_time(max_time=max_times).to(self.device) + 1 # do not sample u=0

        append_time_to_graph_globals(
            batch_to_generate_dense,
            time = self.diffusion_process.normalize_time(u)
        )

        # sample the noisy graph at timestep u

        # WARNING: here selfloops are not masked!!!
        noisy_data = self.diffusion_process.sample_from_original(
            original_datapoint=(batch_to_generate_dense, ext_adjmat),
            t=u
        )

        # onehot and mask the noisy data again (to remove the fake noisy components)
        onehot_data = to_onehot_all(
            *noisy_data,
            **self.data_dims
        )

        # add features to the noisy data
        self.add_additional_features(onehot_data)

        masked_data = mask_all(
            *onehot_data
        )

        noisy_batch_to_generate_dense_onehot, noisy_ext_edges_onehot = masked_data

        #####################  PREDICT THE ORIGINAL GRAPH  #####################
        gen_batch_dense: DenseGraph
        gen_ext_edges: DenseEdges   # None if no external graph
        gen_batch_dense, gen_ext_edges = self.denoising_model(
            graph =                noisy_batch_to_generate_dense_onehot,
            ext_X =                ext_x,
            ext_node_mask =        ext_node_mask,
            ext_edges =            noisy_ext_edges_onehot
        )

        pred_x = gen_batch_dense.x[node_mask]
        pred_e = gen_batch_dense.edge_adjmat[triang_edge_mask]
        if gen_ext_edges is not None:
            pred_ext_e = gen_ext_edges.edge_adjmat[ext_edge_mask]
        else:
            pred_ext_e = None
        pred_c = gen_batch_dense.attribute_node[node_mask]
        pred_dist = gen_batch_dense.attribute_edge[triang_edge_mask]
            
        ###########################  DISTANCE PREDICTION  ############################
        
        ###########################  FINAL PACKING  ############################

        true_values = [true_x, true_e, true_ext_e, true_dist, true_c]
        pred_values = [pred_x, pred_e, pred_ext_e, node_mask, triang_edge_mask, ext_edge_mask, pred_dist, pred_c]
        
        return true_values, pred_values
    

    @torch.no_grad()
    def compute_metrics(
            self,
            loss_logs: Dict[str, Tensor],
            pred_values: List[Tensor],
            true_values: List[Tensor],
            split: str
        ):
        
        metrics = self.metrics[split]

        metrics[labels.DENOISE_CE_X](loss_logs[labels.DENOISE_CE_X])
        metrics[labels.DENOISE_CE_E](loss_logs[labels.DENOISE_CE_E])
        metrics[labels.DENOISE_MSE_DIST](loss_logs[labels.DENOISE_MSE_DIST])
        metrics[labels.DENOISE_TOTAL](loss_logs[labels.DENOISE_TOTAL])
        if pred_values[0].numel() > 0:
            metrics[labels.DENOISE_ACC_X](pred_values[0], true_values[0])
        if pred_values[1].numel() > 0:
            metrics[labels.DENOISE_ACC_E](pred_values[1], true_values[1])
        if pred_values[7].numel() > 0:
            metrics[labels.DENOISE_ACC_C](pred_values[7], true_values[4])
        if pred_values[6].numel() > 0:
            pred_dist_bond = torch.mul(pred_values[6],(torch.argmax(pred_values[1],1)!=0).int())
            true_dist_bond = torch.mul(true_values[3],(torch.argmax(pred_values[1],1)!=0).int())
            metrics[labels.DENOISE_MAE_DIST](pred_dist_bond, true_dist_bond)

        if pred_values[2] is not None:
            metrics[labels.DENOISE_CE_EXT_E](loss_logs[labels.DENOISE_CE_EXT_E])
            if pred_values[2].numel() > 0:
                metrics[labels.DENOISE_ACC_EXT_E](pred_values[2], true_values[2])

        return metrics



    def prepare_batch(self, batch: SparseGraph):

        if self.received_dims['y'] == 0:
            batch.y = None

        return batch
    

    ############################################################################
    #                          TRAINING PHASE SECTION                          #
    ############################################################################

    def on_train_epoch_start(self) -> None:
        self.start_time = time.time()

    def on_train_epoch_end(self) -> None:
        """"Recall that this method is called AFTER the validation epoch, if there is any!"""
        
        denoise_logs = self.apply_prefix(
            metrics = self.metrics[KEY_TRAIN],
            prefix = f'train_denoising'
        )
        self.log_dict(denoise_logs)

        self.total_elapsed_time += time.time() - self.start_time
        self.max_memory_reserved = max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)


    def training_step(self, batch: SparseGraph|Dict, batch_idx: int):

        # compute true and predicted nodes and edges from the denoising process
        if isinstance(batch, dict):

            curr_batch = self.prepare_batch(batch['curr'])
            ext_batch = self.prepare_batch(batch['ext'])
            ext_edges = batch['edges_curr_ext']

            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = curr_batch,
                batch_external =	ext_batch,
                edges_external =	ext_edges,
                train_step=True
            )
            
        else:
            batch = self.prepare_batch(batch)

            # here i will introduce a model that learns distances from moleular graphs
            
            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = batch,
                train_step=True
            )

        # compute denoising training loss
        denoise_loss, denoise_logs = self.train_loss(
            pred_data,
            true_data,
            ret_log=True
        )

        # compute metrics
        self.compute_metrics(denoise_logs, pred_data, true_data, split=KEY_TRAIN)

        # apply prefix to logs
        logs = self.apply_prefix(
            metrics = self.metrics[KEY_TRAIN],
            prefix = f'train_denoising'
        )

        self.log_dict(logs)

        return {'loss': denoise_loss}


    def configure_optimizers(self):

        # currently using the AdamW optimizer
        # NOTE: the original code used the option "amsgrad=True"
        return torch.optim.AdamW(
            self.denoising_model.parameters(), **self.optimizer_config
        )
    
    ############################################################################
    #                         VALID/TEST PHASE SECTION                         #
    ############################################################################

    @torch.no_grad()
    def on_evaluation_epoch_start(self, which=KEY_VALID) -> None:

        # part used for gathering conditioning
        # attributes from the validation or test set
        # to be used for generation
        self.conditioning_y = None
        if self.is_conditional():
            self.conditioning_y = []
            self.num_cond_y = 0


    @torch.no_grad()
    def evaluation_step(self, batch: SparseGraph, batch_idx: int, which=KEY_VALID) -> None:

        batch = self.prepare_batch(batch)

        #############  SAVE PROPERTIES FOR CONDITIONAL GENERATION  #############
        # save some target properties if needed for conditional generation
        if self.is_conditional():

            # get how many will be sampled
            sampling_metrics = self.losses['sampling']
            if which in sampling_metrics:
                sampling_metrics = sampling_metrics[which]

            num_to_sample = sampling_metrics.generation_cfg['num_samples']

            # get the conditioning attributes from the batch
            if self.num_cond_y < num_to_sample:
                to_grab = min(num_to_sample - self.num_cond_y, batch.num_graphs)
                self.conditioning_y.append(batch.y[:to_grab, -2:].float())
                self.num_cond_y += to_grab

        #######################  TRAIN DENOISING MODEL  ########################

        # FLOW DEFINITION
        # survived graph -> encoded survived graph
        # removed graph -> noisy graph -> denoised graph

        # compute true and predicted nodes and edges from the denoising process
        if isinstance(batch, dict):

            curr_batch = self.prepare_batch(batch['curr'])
            ext_batch = self.prepare_batch(batch['ext'])
            ext_edges = batch['edges_curr_ext']

            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = curr_batch,
                batch_external =	ext_batch,
                edges_external =	ext_edges
            )
        else:
            true_data, pred_data = self.compute_true_pred_denoising(
                batch_to_generate = batch
            )

        # compute denoising training loss
        denoise_loss, denoise_logs = self.train_loss(
            pred_data,
            true_data,
            reduce=False,
            ret_log=True
        )

        # compute metrics
        self.compute_metrics(denoise_logs, pred_data, true_data, split=which)

        logs = self.apply_prefix(
            metrics = self.metrics[KEY_VALID],
            prefix = f'valid_denoising'
        )

        self.log_dict(logs)

        return {'loss': denoise_loss}


    @torch.no_grad()
    def on_evaluation_epoch_end(self, which=KEY_VALID) -> None:

        # compute sampling metrics
        do_assignment = self.test_assignment is not None and (
                which == KEY_TEST or
                (which == KEY_VALID and self.validation_config['do_assignment'])
            )


        if do_assignment:

            ######## compute the sampling metrics ########

            if which == KEY_TEST:
                num_samples = self.test_assignment.how_many_to_generate
            else:
                num_samples = self.validation_config.how_many_to_generate

            sampling_data, num_nodes_hist = self.compute_sampling_metrics(
                num_samples = num_samples,
                batch_size = self.generation_config['batch_size']
            )

            ##############################################

            if num_nodes_hist and isinstance(self.logger, WandbLogger):
                # also add the histogram of number of nodes
                wandb.log({f'{which}_sampling/num_nodes_hist': wandb.Histogram(np_histogram=num_nodes_hist)})

            # log computational metrics
            overtime = 0 if self.start_time is None else time.time() - self.start_time
            
            computational_metrics = {
                m_list.KEY_COMPUTATIONAL_TRAIN_TIME: self.total_elapsed_time + overtime,
                m_list.KEY_COMPUTATIONAL_TRAIN_MEMORY:  max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)
            }
            
            # perform assignment
            to_log = self.test_assignment(
                **computational_metrics,
                **self.metrics[which],
                **sampling_data
            )


        else:
            to_log = self.metrics[which]


        self.console_logger.info(str(to_log))

        # add prefix to logs
        to_log = self.apply_prefix(
            metrics = to_log,
            prefix = f'{which}'
        )

        self.log_dict(to_log)


    @torch.no_grad()
    def compute_sampling_metrics(self, num_samples, batch_size: int=64):

        self.console_logger.info('Sampling some graphs...')

        if self.is_conditional():
            conditioning_y = torch.cat(self.conditioning_y, dim=0)
            num_available_y = conditioning_y.shape[0]
            if num_available_y < num_samples:
                self.console_logger.warning(f'Only {num_available_y} conditioning attributes available, but {num_samples} requested')
                self.console_logger.info('Sampling with replacement...')
                conditioning_y_nolast = conditioning_y.repeat(num_samples // num_available_y, 1)
                if num_samples % num_available_y > 0:
                    conditioning_y = torch.cat([conditioning_y_nolast, conditioning_y[:num_samples % num_available_y]], dim=0)
                else:
                    conditioning_y = conditioning_y_nolast

            # split into the batches to generate
            conditioning_y = torch.split(conditioning_y, batch_size)
        else:
            conditioning_y = [None] * (num_samples // batch_size + 1)
        
        # initialize for process metrics
        torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()

        # sample required graphs
        samples = self.sample(
            num_samples = num_samples,
            condition = conditioning_y,
            batch_size = batch_size
        )

        # end for process metrics
        end_time = time.time()
        peak_memory_usage = float(torch.cuda.max_memory_allocated(0))

        self.console_logger.info(f'Done. Sampling took {end_time - start_time:.2f} seconds\n')

        # compute some statistics on the generated graphs
        num_nodes = [s.num_nodes for s in samples]
        num_edges = [s.num_edges for s in samples]
        self.console_logger.info(f'Number of nodes per graph: avg:{np.mean(num_nodes)}, min:{np.min(num_nodes)}, max:{np.max(num_nodes)}')
        self.console_logger.info(f'Number of edges per graph: avg:{np.mean(num_edges)}, min:{np.min(num_edges)}, max:{np.max(num_edges)}')

        # compute histogram of number of nodes
        num_nodes_hist = np.histogram(num_nodes, bins=np.arange(min(num_nodes)-0.5, max(num_nodes)+1.5), density=True)

        sampling_data = {
            'data': samples,
            'time': {'start': start_time, 'end': end_time},
            'memory': {'peak': peak_memory_usage}
        }

        return sampling_data, num_nodes_hist


    ############################################################################
    #           VALIDATION PHASE SECTION (executed during validation)          #
    ############################################################################

    def on_validation_epoch_start(self):
        self.on_evaluation_epoch_start(which=KEY_VALID)

    def validation_step(self, batch: SparseGraph, batch_idx: int):
        return self.evaluation_step(batch, batch_idx, which=KEY_VALID)

    def on_validation_epoch_end(self):
        return self.on_evaluation_epoch_end(which=KEY_VALID)

    ############################################################################
    #               TEST PHASE SECTION (executed during testing)               #
    ############################################################################

    def on_test_epoch_start(self):
        self.on_evaluation_epoch_start(which=KEY_TEST)

    def test_step(self, batch: SparseGraph, batch_idx: int):
        return self.evaluation_step(batch, batch_idx, which=KEY_TEST)

    def on_test_epoch_end(self):
        return self.on_evaluation_epoch_end(which=KEY_TEST)

    ############################################################################
    #                           CHECKPOINT FUNCTIONS                           #
    ############################################################################


    def apply_prefix(self, metrics, prefix):
        return {f'{prefix}/{k}'.lower(): v for k, v in metrics.items()}


    def log_wandb_media(self, name, metric):
        wandb.log({name: metric})


    ############################################################################
    #                           MODEL CALL FUNCTIONS                           #
    ############################################################################
    
    @torch.no_grad()
    def forward_denoising(
            self,
            graph_to_gen: DenseGraph,
            ext_edges_to_gen: DenseEdges,
            encoded_ext_x: Tensor,
            ext_node_mask: Tensor,
            denoising_time: IntTensor,
            return_onehot: bool=True,
            return_masked: bool=True,
            copy_globals_to_output: bool=True
        ) -> Tuple[DenseGraph, Tensor]:	

        #assert_is_onehot(graph_to_gen, ext_edges_to_gen)

        augmented_graph_to_gen = graph_to_gen.clone()
        self.add_additional_features((augmented_graph_to_gen, ext_edges_to_gen))


        # predict final graph and edges
        final_graph: DenseGraph
        final_ext_edges: DenseEdges
        final_graph, final_ext_edges = self.denoising_model(
            graph =				    augmented_graph_to_gen,
            ext_X =					encoded_ext_x,
            ext_node_mask =			ext_node_mask,
            ext_edges =         	ext_edges_to_gen
        )

        has_ext_edges = final_ext_edges is not None
        
        # transform the logits to probabilities
        final_graph.x = torch.softmax(final_graph.x, dim=-1)
        #final_graph.edge_adjmat = torch.softmax(final_graph.edge_adjmat, dim=-1)
        value_sigmoid = torch.sigmoid(final_graph.edge_adjmat[:,:,:,0])
        value_softmax = torch.softmax(final_graph.edge_adjmat[:,:,:,1:], dim=-1)
        final_graph.edge_adjmat=torch.cat((value_sigmoid.unsqueeze(3), (1-value_sigmoid).unsqueeze(3)*value_softmax), dim=-1)
        final_graph.attribute_node = torch.softmax(final_graph.attribute_node, dim=-1)
        if has_ext_edges:
            final_ext_edges.edge_adjmat = torch.softmax(final_ext_edges.edge_adjmat, dim=-1)
        else:
            final_ext_edges = None
        

        # pack datapoints
        original_datapoint = (final_graph, final_ext_edges)
        current_datapoint = (graph_to_gen, ext_edges_to_gen)

        # sample graph at step t-1 from posterior
        generated_graph, generated_ext_edges = self.diffusion_process.sample_posterior(
            original_datapoint =	original_datapoint,
            current_datapoint =		current_datapoint,
            t =						denoising_time
        )

        if return_onehot:
            generated_graph, generated_ext_edges = to_onehot_all(
                generated_graph, generated_ext_edges,
                **self.data_dims
            )

        if return_masked:
            generated_graph, generated_ext_adjmat = mask_all(
                generated_graph, generated_ext_edges
            )


        if copy_globals_to_output:
            generated_graph.y = graph_to_gen.y

        return generated_graph, generated_ext_adjmat

    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        conditioning_y: Optional[Tensor]=None,
        ext_graph: Optional[SparseGraph]=None,
        encoded_ext_x: Optional[Tensor]=None,
        number_of_nodes: Optional[IntTensor]=None,
        return_directed: bool=True,
        save_chains: int=0
    ):
        ########################################################################
        #                        INITIAL SAMPLING SETUP                        #
        ########################################################################

        #########################  SETUP CONDITIONING  #########################

        # TODO: implement the generation chain saving
        do_save_chains = save_chains > 0


        # elaborate external graph
        # if it is None, then all parts about it will be skipped
        if ext_graph is not None:
            encoded_ext_x, ext_node_mask = to_dense_batch(
                x =				encoded_ext_x,
                batch =			ext_graph.batch,
                batch_size =	ext_graph.num_graphs
            )
        else:
            encoded_ext_x = ext_node_mask = None

        # if the number of nodes is not given, sample it from the empirical distribution
        if number_of_nodes is None:
            number_of_nodes = self.empirical_sampler(batch_size)

        ##############  SAMPLE THE STARTING SUBGRAPHS (AS NOISE)  ##############
        new_graph: DenseGraph
        new_ext_edges: DenseEdges
        new_graph, new_ext_edges = self.diffusion_process.sample_stationary(
            num_new_nodes = number_of_nodes,
            ext_node_mask = ext_node_mask,
            num_classes = self.data_dims # initialize with true data dimensions
        )

        # convert the new subgraph to one-hot
        new_graph, new_ext_edges = to_onehot_all(
            *(new_graph, new_ext_edges),
            **self.data_dims
        )

        # copy the global information to the new subgraph
        if ext_graph is not None:
            new_graph.y = ext_graph.y.clone()
        elif conditioning_y is not None:
            new_graph.y = conditioning_y.clone()
        else:
            new_graph.y = None
        
        ###################  INITIALIZE DENOISING TIME AS T  ###################
        diffusion_max_time = self.diffusion_process.get_max_time()

        diff_time = self.diffusion_process.normalize_time(
            t = torch.full((batch_size,), diffusion_max_time, dtype=torch.int, device=self.device)
        )

        append_time_to_graph_globals(
            graph = new_graph,
            time = diff_time
        )

        new_graph_dense = new_graph

        ########################################################################
        #                            DENOISING LOOP                            #
        ########################################################################

        t_tensor = torch.empty(batch_size, dtype=torch.int, device=self.device)

        # denoise going backwards in time
        for t in reversed(range(1,diffusion_max_time+1)):

            t_tensor.fill_(t)

            # sample graph at step u-1
            new_graph_dense, new_ext_edges = self.forward_denoising(
                graph_to_gen =		new_graph_dense,
                ext_edges_to_gen =	new_ext_edges,
                encoded_ext_x =		encoded_ext_x,
                ext_node_mask =		ext_node_mask,
                denoising_time = 	t_tensor,
                return_onehot =		True
            )

            # update denoising time (in-place), denoising go down!
            new_graph_dense.y[..., 0] = self.diffusion_process.normalize_time(
                t = t-1
            )

        #######################  END OF DENOISING LOOP  ########################

        ########################################################################
        #                    MERGE THE OLD AND NEW SUBGRAPHS                   #
        ########################################################################

        # this operation only sparsify new_graph_dense if external graph
        # is None
        output_graph = merge_sparse_dense_graphs_to_sparse(
            sparse_subgraph = ext_graph,
            dense_subgraph = new_graph_dense,
            dense_ext_edges = new_ext_edges,
            dense_nodes_num = number_of_nodes
        )

        ########################################################################
        #                                RETURN                                #
        ########################################################################

        # replace globals with starting variables, removing time
        if ext_graph is not None:
            output_graph.y = ext_graph.y
        elif conditioning_y is not None:
            output_graph.y = conditioning_y
        else:
            output_graph.y = None
        
        return output_graph
            
    
    @torch.no_grad()
    def sample(
            self,
            num_samples: int,
            condition: Optional[Dict]=None,
            batch_size: Optional[int]=None
        ):

        if batch_size is None:
            batch_size = self.generation_config['batch_size']

        samples_left_to_generate = num_samples
        batch_idx = 0
        samples = []

        while samples_left_to_generate > 0:
            to_generate = min(samples_left_to_generate, batch_size)
            self.console_logger.info(f'Generating {to_generate} graphs...')

            graph_batch = self.sample_batch(
                batch_size=to_generate,
                conditioning_y=condition[batch_idx]
            )

            graph_batch.collapse()
            
            graph_batch.attribute_node = torch.argmax(graph_batch.attribute_node, dim = -1)-1
            
            output_batch = graph_batch.to_data_list()

            output_batch = [g.cpu() for g in output_batch]

            samples.extend(output_batch)

            samples_left_to_generate -= to_generate
            batch_idx += 1
            self.console_logger.info(f'Generated {len(samples)}/{num_samples} graphs')

        return samples
    

    ############################################################################
    #                         UTILITY MODULE FUNCTIONS                         #
    ############################################################################


    def add_additional_features(self, graph: SparseGraph|DenseGraph|Tuple[DenseGraph, DenseEdges]) -> Tensor:

        for feature in self.additional_features:
            feature(graph)

        return graph


################################################################################
#                               UTILITY METHODS                                #
################################################################################

# the following methods are utility methods which could be an integral part of
# the main class, but have been put outside for readability

##############################  DATA FORMATTING  ###############################

def format_generation_task_data(
        curr_graph: SparseGraph,
        ext_graph: SparseGraph,
        edges_curr_ext: SparseEdges=None,
        edges_ext_curr: SparseEdges=None
    ) -> Tuple[DenseGraph, Tensor, BoolTensor, DenseEdges, DenseEdges]:
    """transform the splitting of the two graphs into the format required by the
    model, that is:
    - extract a dense representation (and a node mask) of the Ne nodes from ext_graph
    - transform curr_graph into a DenseGraph (with Nc nodes, and adjmat of shape (*, Nc, Nc, *))
    - transform edges_ext_curr and edges_curr_ext into dense adjacency matrices
      each of shape (*, Ne, Nc, *) and (*, Nc, Ne, *) respectively.
      If one of the two is None, it is assumed that the graph is undirected and
      a single adjacency matrix ((*, Ne, Nc, *) or (*, Nc, Ne, *)) is returned.

    Notice that the possibly very big adjacency matrix of curr_graph (*, Nc, Nc, *)
    is never computed, so Nc >> Ne is allowed, avoiding a squared dependency on
    Nc.

    Parameters
    ----------
    curr_graph : SparseGraph
        graph of nodes in the current graph to be generated
    ext_graph : SparseGraph
        graph of nodes from an external graph
    edges_ext_curr : Tuple[Tensor, Tensor]
        edges going from the external nodes to the current nodes. The first
        component is the edge_index, the second the edge_attr. If is is None,
        the dense version is not returned (default: None)
    edges_curr_ext : Tuple[Tensor, Tensor], optional
        edges going from the current nodes to the external nodes. The first
        component is the edge_index, the second the edge_attr. If is is None,
        the dense version is not returned (default: None)

    Returns
    -------
    curr_graph_dense : DenseGraph
        graph of nodes removed by the removal process as a dense graph.
    ext_x_tensor : Tensor
        tensor of the external nodes, as a batched dense representation.
    ext_node_mask : BoolTensor
        mask of the true external nodes, as the process of densifying generates
        some dummy nodes.
    adjmat_ext_curr : Optional[Tensor]
        edges going from the external nodes to the current nodes, as a dense
        adjacency matrix. If edges_ext_curr is None, this is not returned.
    adjmat_curr_ext : Optional[Tensor]
        edges going from the current nodes to the external nodes, as a dense
        adjacency matrix. If edges_curr_ext is None, this is not returned.
    """

    batch_size = curr_graph.num_graphs

    # transform the current graph into a dense representation
    curr_graph_dense = dense.sparse_graph_to_dense_graph(
        sparse_graph =		curr_graph,
        handle_one_hot =    True
    )

    # reduce the padding to the current padding size

    curr_graph_dense.attribute_edge=curr_graph_dense.attribute_edge[:, 0:curr_graph_dense.x.shape[1], 0:curr_graph_dense.x.shape[1]]
    
    try:
        curr_graph_dense.pos = curr_graph_dense.pos[:, 0:curr_graph_dense.x.shape[1], :]
    except TypeError:
        pass
    
    try:    
        curr_graph_dense.attribute_node=curr_graph_dense.attribute_node[:, 0:curr_graph_dense.x.shape[1]]
    except TypeError:
        pass
    

    # initialize to None
    ext_x_tensor = None
    ext_node_mask = None
    edges_ext_curr_dense = None
    edges_curr_ext_dense = None

    # if conditioned on an external graph
    if ext_graph is not None:

        # extract the dense representation of the surviving nodes
        ext_x_tensor, ext_node_mask = to_dense_batch(
            x =             ext_graph.x,
            batch =         ext_graph.batch,
            batch_size =    batch_size
        )

        if (edges_ext_curr is not None) or (edges_curr_ext is not None):
            edge_mask_ext_curr = dense.get_bipartite_edge_mask_dense(
                node_mask_a = ext_node_mask,
                node_mask_b = curr_graph_dense.node_mask
            )

        # transform the edges_curr_ext into a dense adjacency matrix
        if edges_curr_ext is not None:
            # transpose
            edge_mask_curr_ext = edge_mask_ext_curr.transpose(1, 2)

            adjmat_curr_ext = dense.to_dense_adj_bipartite(
                edge_index =	edges_curr_ext.edge_index,
                edge_attr =		edges_curr_ext.edge_attr,
                batch_s =		curr_graph.batch,
                batch_t =		ext_graph.batch,
                batch_size =	batch_size,
                handle_one_hot =True,
                edge_mask =     edge_mask_curr_ext
            )

            edges_curr_ext_dense = DenseEdges(
                edge_adjmat =   adjmat_curr_ext,
                edge_mask =     edge_mask_curr_ext
            )

        # transform the edges_ext_curr into a dense adjacency matrix
        if edges_ext_curr is not None:

            adjmat_ext_curr =   dense.to_dense_adj_bipartite(
                edge_index =    edges_ext_curr.edge_index,
                edge_attr =     edges_ext_curr.edge_attr,
                batch_s =       ext_graph.batch,
                batch_t =       curr_graph.batch,
                batch_size =    batch_size,
                handle_one_hot =True,
                edge_mask =     edge_mask_ext_curr
            )

            edges_ext_curr_dense = DenseEdges(
                edge_adjmat =   adjmat_ext_curr,
                edge_mask =     edge_mask_ext_curr
            )
            

    return curr_graph_dense, ext_x_tensor, ext_node_mask, edges_curr_ext_dense, edges_ext_curr_dense



def sparsify_data(
        subgraph: DenseGraph,
        ext_edges: DenseEdges,
        subgraph_nodes_num: IntTensor,
        ext_ptr: Tensor = None,
    ) -> Tuple[SparseGraph, SparseEdges]:

    ########################  SPARSIFY DENSE SUBGRAPH  #########################
    subgraph = subgraph.clone()

    # remove self-loops from dense adjacency matrices
    subgraph.edge_adjmat = dense.dense_remove_self_loops(
        subgraph.edge_adjmat
    )

    # remove no edge class from dense adjacency
    # matrices
    subgraph.edge_adjmat = dense.remove_no_edge(
        subgraph.edge_adjmat,
        sparse = False,
        collapsed = False
    )

    # transform the new graph to sparse format
    new_subgraph = dense.dense_graph_to_sparse_graph(
        dense_graph =	subgraph,
        num_nodes =		subgraph_nodes_num,
        batchify =      True
    )

    ##########################  SPARSIFY DENSE EDGES  ##########################

    if ext_edges is not None:

        ext_edges.edge_adjmat = dense.remove_no_edge(
            ext_edges.edge_adjmat,
            sparse = False,
            collapsed = False
        )

        new_edges = dense.dense_edges_to_sparse_edges(
            dense_edges =		ext_edges,
            cum_num_nodes_s =	new_subgraph.ptr,
            cum_num_nodes_t =	ext_ptr
        )

    else:
        new_edges = None

    return new_subgraph, new_edges


def merge_sparse_dense_graphs_to_sparse(
        sparse_subgraph: SparseGraph,
        dense_subgraph: DenseGraph,
        dense_ext_edges: DenseEdges,
        dense_nodes_num: IntTensor
    ) -> SparseGraph:

    # sparsify the dense graph
    new_subgraph, new_edges_ba = sparsify_data(
        subgraph =				dense_subgraph,
        ext_edges =				dense_ext_edges,
        subgraph_nodes_num =	dense_nodes_num,
        ext_ptr = 				sparse_subgraph.ptr if sparse_subgraph is not None else None
    )

    if new_edges_ba is not None:

        # get both directions of the new edges
        new_edges_ab = new_edges_ba.clone().transpose()

        # merge the sparse graph with the sparsified dense graph
        merged_graph = split.merge_subgraphs(
            graph_a =	sparse_subgraph,
            graph_b =	new_subgraph,
            edges_ab =	new_edges_ab,
            edges_ba =	new_edges_ba
        )

    else:
        merged_graph = new_subgraph

    return merged_graph


###########################  BULK OPERATION METHODS  ###########################

def to_onehot_all(*data, **classes_nums):

    ret_data = []

    for i, d in enumerate(data):
        if isinstance(d, tuple):
            k, d = d
            ret_d = F.one_hot(
                d.long(), num_classes = classes_nums[k]
            ).float()

        elif isinstance(d, DenseEdges):
            ret_d = d.to_onehot(
                num_classes_e =	classes_nums['e']
            )
        
        elif isinstance(d, (DenseGraph, SparseGraph)):
            ret_d = d.to_onehot(
                num_classes_x =	classes_nums['x'],
                num_classes_e =	classes_nums['e'],
                num_classes_c = classes_nums['c'],
            )

        elif isinstance(d, Tensor):
            if d.dtype == torch.bool:
                ret_d = d.unsqueeze(-1)

        elif d is None:
            ret_d = None

        else:
            raise NotImplementedError(f'{i}-th data of type {type(d)} during to_onehot_all')
        
        ret_data.append(ret_d)

    return ret_data


def mask_all(*data, **masks):

    ret_data = []

    for i, d in enumerate(data):
        if isinstance(d, tuple):
            k, d = d
            ret_d = d * masks[k].unsqueeze(-1)
        
        elif isinstance(d, DenseGraph):
            ret_d = d.apply_mask()

        elif d is None:
            ret_d = None

        else:
            raise NotImplementedError(f'{i}-th data of type {type(d)} during mask_all')

        ret_data.append(ret_d)

    return ret_data


#########################  SMALL REPEATED OPERATIONS  ##########################
# the following methods are meant to abstract away some small operations that
# are repeated in the code

def append_time_to_graph_globals(
        graph: Union[DenseGraph, SparseGraph],
        time: Union[IntTensor, LongTensor],
    ) -> Union[DenseGraph, SparseGraph]:
    """Append the time to the graph globals vector y
    with the following criteria:
    - if the graph has no y, set y = time
    - if the graph has y, set y = [time, y], that is,
        the time is appended to the beginning of the vector

    Parameters
    ----------
    graph : Union[DenseGraph, SparseGraph]
        any kind of graph with batched y vector of size [batch_size, *] or None
    time : Union[IntTensor, LongTensor]
        time tensor of size [batch_size], this method will unsqueeze to [batch_size, 1]

    Returns
    -------
    same_graph : Union[DenseGraph, SparseGraph]
        same graph as the input, but with the updated y vector
    """

    time = time.float().unsqueeze(-1)

    if graph.y is None:
        graph.y = time
    else:
        if graph.y.ndim == 1:
            graph.y = graph.y.unsqueeze(-1)
        graph.y = torch.cat([time, graph.y], dim = -1)

    return graph





#################################  ASSERTIONS  #################################

def assert_is_onehot(*data):

    tensor_dims = {
        'xd': ('dense nodes', 3),
        'xs': ('sparse nodes', 2),
        'ed': ('dense edges', 4),
        'es': ('sparse edges', 2)
    }

    for i, d in enumerate(data):
        if isinstance(d, tuple):

            k: str
            d: Tensor
            k, d = d
            
            assert d.ndim == tensor_dims[k][1], \
                f'Expected {tensor_dims[k][0]} to be of dimension {tensor_dims[k][1]}, got {d.ndim}'

        elif isinstance(d, DenseGraph):
            assert not d.collapsed, \
                'Expected the dense graph to be onehot'
        
        elif isinstance(d, SparseGraph):
            assert_is_onehot(
                ('xs', d.x),
                ('es', d.edge_attr)
            )

        else:
            raise NotImplementedError(f'Expected {i}-th data to be of type tuple, DenseGraph or SparseGraph, got {type(d)}')
            
            
