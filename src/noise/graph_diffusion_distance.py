from typing import Tuple, Dict

import numpy as np

import torch
from torch import Tensor, IntTensor, BoolTensor
from torch_geometric.utils.to_dense_batch import to_dense_batch

from src.models.architectures.gnn.distance_graph_transformer_qm9 import DIM_X, DIM_E, DIM_C, DIM_DIST

from src.datatypes.dense import (
    DenseGraph,
    DenseEdges,
    get_bipartite_edge_mask_dense,
    get_edge_mask_dense
)

from src.noise import NoiseSchedule, NoiseProcess

################################################################################
#                              UTILITY FUNCTIONS                               #
################################################################################



class DiffusionProcessException(Exception):
    pass


def cosine_beta_schedule_discrete(max_steps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = max_steps + 1
    x = np.linspace(0, max_steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / max_steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = np.concatenate([np.ones(1), alphas_cumprod[1:] / alphas_cumprod[:-1]])
    betas = 1 - alphas
    return betas.squeeze()


def time_to_long(t: Tensor, timesteps: int):
    if t.dtype == torch.long:
        return t

    elif t.dtype == torch.int:
        return t.long()
    
    elif t.dtype == torch.float:
        t_int = torch.round(t * timesteps)
        return t_int.long()
    
    else:
        raise DiffusionProcessException(
            f'Given time tensor t has wrong dtype: {t.dtype}. Should be long, integer or float in [0,1]'
        )


################################################################################
#                         DIFFUSION PROCESS SCHEDULES                          #
################################################################################

class CosineDiffusionSchedule_distance(NoiseSchedule):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, max_time: int):
        super().__init__()

        self.max_time = max_time

        # compute betas (parameter next)
        betas = cosine_beta_schedule_discrete(max_time)
        # clamp values as in the original paper
        betas = torch.clamp(torch.from_numpy(betas), min=0, max=0.999)
        self.register_buffer('betas', betas.float())

        # compute alpha = 1 - beta
        alphas = 1 - self.betas

        # recompute alpha_bar (parameter time 0->t)
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        
        self._alphas_bar = torch.exp(log_alpha_bar)
        
        self.log_alpha_bar = log_alpha_bar
        
        self._sigma2_bar = -torch.expm1(2 * log_alpha_bar)
        self._sigma_bar = torch.sqrt(self._sigma2_bar)
        self.register_buffer('alphas_bar', torch.exp(log_alpha_bar))


    def params_next(self, t: Tensor, **kwargs):
        t_int = time_to_long(t, self.max_time)

        return self.betas[t_int]

    def params_time_t(self, t: Tensor, **kwargs):
        t_int = time_to_long(t, self.max_time)

        return self.alphas_bar[t_int]

    def get_sigma_bar(self, t_normalized=None, t_int=None, key=None):
        #assert int(t_normalized is None) + int(t_int is None) == 1
        #if t_int is None:
        #    t_int = torch.round(t_normalized * self.T)
        s = self._sigma_bar.to(t_int.device)[t_int]
        
        return s.float()
    
    def get_alpha_pos_ts(self, t_int, s_int):
        log_a_bar = self.log_alpha_bar.to(t_int.device)
        ratio = torch.exp(log_a_bar[t_int] - log_a_bar[s_int])
        return ratio.float()

    def get_alpha_pos_ts_sq(self, t_int, s_int):
        log_a_bar = self.log_alpha_bar.to(t_int.device)
        ratio = torch.exp(2 * log_a_bar[t_int] - 2 * log_a_bar[s_int])
        return ratio.float()
    
    def get_alpha_bar(self, t_normalized=None, t_int=None):
        #assert int(t_normalized is None) + int(t_int is None) == 1
        #if t_int is None:
        #    t_int = torch.round(t_normalized * self.T)
        a = self._alphas_bar.to(t_int.device)[t_int.long()]
        return a.float()

    def get_sigma_pos_sq_ratio(self, s_int, t_int):
        log_a_bar = self.log_alpha_bar.to(t_int.device)
        s2_s = - torch.expm1(2 * log_a_bar[s_int])
        s2_t = - torch.expm1(2 * log_a_bar[t_int])
        ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t))
        return ratio.float()

    def get_x_pos_prefactor(self, s_int, t_int):
        """ a_s (s_t^2 - a_t_s^2 s_s^2) / s_t^2"""
        a_s = self.get_alpha_bar(t_int=s_int)
        alpha_ratio_sq = self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
        sigma_ratio_sq = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        prefactor = a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)
        return prefactor.float()

    def params_posterior(self, t, **kwargs):
        raise NotImplementedError

    def get_max_time(self, **kwargs):
        return self.max_time

    def forward(self, t: Tensor, **kwargs):
        return self.params_next(t)

################################################################################
#                             DIFFUSION PROCESSES                              #
################################################################################


# adapted from https://github.com/cvignac/DiGress/blob/main/src/diffusion/diffusion_utils.py

def remove_mean_with_mask(x, node_mask):
    """ x: bs x n x d.
        node_mask: bs x n """
    assert node_mask.dtype == torch.bool, f"Wrong type {node_mask.dtype}"
    node_mask = node_mask.unsqueeze(-1)
    masked_max_abs_value = (x * (~node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def compute_matmul_graph(
        datapoint: Tuple[DenseGraph, DenseEdges],
        noise_graph: DenseGraph
    ) -> Tuple[DenseGraph, DenseEdges]:

    # unpack datapoint
    graph, ext_edges = datapoint
    with_external = ext_edges is not None
    
    # apply noise to own nodes, edges, and edges leading to external graph
    prob_x = graph.x @ noise_graph.x									# (bs, nq, dx_out)
    prob_e = graph.edge_adjmat @ noise_graph.edge_adjmat.unsqueeze(1)	# (bs, nq, nq, de_out)
    prob_ext_e = ext_edges.edge_adjmat @ noise_graph.edge_adjmat.unsqueeze(1) \
        if with_external else None # (bs, nq, nk, de_out)
    prob_c = graph.attribute_node @ noise_graph.attribute_node
    
    # prepare probability graph
    prob_graph = DenseGraph(
        x =				prob_x,
        edge_adjmat =	prob_e,
        y =				graph.y,
        attribute_edge=      noise_graph.attribute_edge,
        attribute_node=         prob_c,
        node_mask =		graph.node_mask,
        edge_mask =     graph.edge_mask,
    )

    prob_ext_edges = DenseEdges(
        edge_adjmat =   prob_ext_e,
        edge_mask =     ext_edges.edge_mask
    ) if with_external else None

    return (prob_graph, prob_ext_edges)


# non modificato perchè non usato
def compute_elementwise_graph(
        first_datapoint: Tuple[DenseGraph, DenseEdges],
        second_datapoint: Tuple[DenseGraph, DenseEdges],
    ) -> Tuple[DenseGraph, DenseEdges]:

    # unpack datapoint
    first_graph, first_ext_edges = first_datapoint
    second_graph, second_ext_edges = second_datapoint

    with_external = first_ext_edges is not None
    
    # apply noise to own nodes, edges, and edges leading to external graph
    prob_x = first_graph.x * second_graph.x							# (bs, nq, dx_out)
    prob_e = first_graph.edge_adjmat * second_graph.edge_adjmat		# (bs, nq, nq, de_out)
    prob_ext_e = first_ext_edges * second_ext_edges	\
        if with_external else None	# (bs, nq, nk, de_out)

    # prepare probability graph
    prob_graph = DenseGraph(
        x =				prob_x,
        edge_adjmat =	prob_e,
        y =				first_graph.y,
        node_mask =		first_graph.node_mask,
        edge_mask =     first_graph.edge_mask
    )

    prob_ext_edges = DenseEdges(
        edge_adjmat =   prob_ext_e,
        edge_mask =     first_ext_edges.edge_mask
    ) if with_external else None

    return (prob_graph, prob_ext_edges)


def compute_prob_s_t_given_0(
        X_t: Tensor,
        Qt: Tensor,
        Qsb: Tensor,
        Qtb: Tensor
    ):
    """Borrowed from https://github.com/cvignac/DiGress/blob/main/src/diffusion/diffusion_utils.py"""

    X_t = X_t.flatten(start_dim=1, end_dim=-2)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out



def normalize_probability(
        x: Tensor,
        norm_x: Tensor
    ) -> Tensor:

    if x is None: return None

    denominator = norm_x.sum(-1, keepdim=True)
    denominator[denominator == 0] = 1

    return x / denominator

# non cambiato perchè non usato
def normalize_graph(
        datapoint: Tuple[DenseGraph, DenseEdges],
        norm_datapoint: Tuple[DenseGraph, DenseEdges],
    ) -> Tuple[DenseGraph, DenseEdges]:

    # gather all relevant tensors
    tuple_numerators = (datapoint[0].x, datapoint[0].edge_adjmat, datapoint[1].edge_adjmat)
    tuple_denominators = (norm_datapoint[0].x, norm_datapoint[0].edge_adjmat, norm_datapoint[1].edge_adjmat)

    # apply normalization
    tuple_normalized = [
        normalize_probability(*tup) for tup in zip(tuple_numerators, tuple_denominators)
    ]

    prob_graph = DenseGraph(
        x =				tuple_normalized[0],
        edge_adjmat =	tuple_normalized[1],
        y =				datapoint[0].y,
        node_mask =		datapoint[0].node_mask,
        edge_mask =     datapoint[0].edge_mask
    )

    prob_ext_edges = DenseEdges(
        edge_adjmat =   tuple_normalized[2],
        edge_mask =     datapoint[1].edge_mask
    ) if datapoint[1] is not None else None

    return prob_graph, prob_ext_edges


def fill_out_prob_graph(
        prob_datapoint: Tuple[DenseGraph, DenseEdges],
    ) -> Tuple[DenseGraph, DenseEdges]:

    # unpack datapoint
    prob_graph, prob_ext_edges = prob_datapoint
    with_external = prob_ext_edges is not None

    # dimensions
    bs, nq, _, _ = prob_graph.edge_adjmat.shape

    num_cls_x = prob_graph.x.shape[-1]
    num_cls_e = prob_graph.edge_adjmat.shape[-1]
    num_cls_c = prob_graph.attribute_node.shape[-1]

    #############  APPLY NOISE TO X  #############
    # Noise X
    # The masked rows should define probability distributions as well
    prob_graph.x[~prob_graph.node_mask] = 1 / num_cls_x

    #############  APPLY NOISE TO E  #############

    # Noise E
    # The masked rows should define probability distributions as well
    diag_mask = torch.eye(nq, dtype=torch.bool, device=prob_graph.device).unsqueeze(0).expand(bs, -1, -1)

    prob_graph.edge_adjmat[~prob_graph.edge_mask] = 1 / num_cls_e # fake nodes
    prob_graph.edge_adjmat[diag_mask] = 1 / num_cls_e # self loops

    #########  APPLY NOISE TO EXTERNAL E  ########
    if with_external:
        # Noise E
        # The masked rows should define probability distributions as well
        prob_ext_edges.edge_adjmat[~prob_ext_edges.edge_mask] = 1 / num_cls_e
        
    ######### APPLY NOISE TO ATTRIBUTE NODE #########
    
    prob_graph.attribute_node[~prob_graph.node_mask] = 1 / num_cls_c

    return (prob_graph, prob_ext_edges)


def sample_from_probabilities(
        prob_datapoint: Tuple[DenseGraph, DenseEdges],
    ) -> Tuple[DenseGraph, DenseEdges]:

    # unpack datapoint
    prob_graph, prob_ext_edges = prob_datapoint
    with_external = prob_ext_edges is not None

    # dimensions
    bs, nq, _, _ = prob_graph.edge_adjmat.shape

    if nq > 0:
        ##############  SAMPLE NODES X  ##############
        # Flatten the probability tensor to sample with multinomial
        prob_x = prob_graph.x.reshape(bs * nq, -1)		# (bs * nq, dx_out)

        # Sample X
        x = prob_x.multinomial(1).reshape(bs, nq)	# (bs, nq)

        #########  SAMPLE INTERNAL EDGES E  ##########
        # Flatten the probability tensor to sample with multinomial
        prob_e = prob_graph.edge_adjmat.reshape(bs * nq * nq, -1)	# (bs * nq * nq, de_out)

        A=(torch.repeat_interleave((prob_e[:,0]>0.5).float(), repeats=4).reshape(len((prob_e[:,0]>0.5).float()),4)).float()   
        A[A==0]=3
        A[A==1]=0
        A[A==3]=1
        # Sample E
        edge_adjmat = prob_e.multinomial(1).reshape(bs, nq, nq)	# (bs, nq, nq)
        edge_adjmat = torch.triu(edge_adjmat, diagonal=1)
        edge_adjmat = edge_adjmat + torch.transpose(edge_adjmat, 1, 2)
        
        # Sample attribute node 
        
        prob_c = prob_graph.attribute_node.reshape(bs*nq, -1)
        c = prob_c.multinomial(1).reshape(bs, nq)

        #########  SAMPLE EXTERNAL EDGES E  ##########
        if with_external:
            bs, nq, nk, _ = prob_ext_edges.edge_adjmat.shape
            if nk == 0:
                ext_edge_adjmat = torch.zeros(bs, nq, nk, dtype=torch.long, device=prob_ext_edges.device)
            else:
                # Flatten the probability tensor to sample with multinomial
                prob_ext_e = prob_ext_edges.edge_adjmat.reshape(bs * nq * nk, -1)	# (bs * nq * nk, de_out)

                # Sample E external
                ext_edge_adjmat = prob_ext_e.multinomial(1).reshape(bs, nq, nk)	# (bs, nq, nk)
    else:
        device = prob_ext_edges.device
        x = torch.zeros(bs, nq, dtype=torch.long, device=device)
        edge_adjmat = torch.zeros(bs, nq, nq, dtype=torch.long, device=device)
        if with_external:
            bs, nq, nk, _ = prob_ext_edges.edge_adjmat.shape
            ext_edge_adjmat = torch.zeros(bs, nq, nk, dtype=torch.long, device=device)
        else:
            ext_edge_adjmat = None

    #############  FORMAT AND RETURN  ############
    # prepare sampled graph and mask
    sampled_graph = DenseGraph(
        x =				x,
        edge_adjmat =	edge_adjmat,
        y =				prob_graph.y,
        node_mask =		prob_graph.node_mask,
        attribute_node=      c,
        attribute_edge=     prob_graph.attribute_edge,
        edge_mask=      prob_graph.edge_mask,
        pos=prob_datapoint[0].pos
    ).apply_mask()

    if with_external:
        # apply mask to external edges
        sampled_ext_edges = DenseEdges(
            edge_adjmat =   ext_edge_adjmat,
            edge_mask =     prob_ext_edges.edge_mask
        ).apply_mask()
    else:
        sampled_ext_edges = None

    return sampled_graph, sampled_ext_edges


def apply_noise_graph(
        datapoint: Tuple[DenseGraph, DenseEdges],
        noise_graph: DenseGraph
    ) -> Tuple[DenseGraph, DenseEdges]:
    
    # get transition probabilities
    prob_datapoint = compute_matmul_graph(
        datapoint=datapoint,
        noise_graph=noise_graph
    )

    # fill out probability graph
    prob_datapoint = fill_out_prob_graph(
        prob_datapoint=prob_datapoint
    )

    # sample from transition probabilities
    sampled_datapoint = sample_from_probabilities(
        prob_datapoint=prob_datapoint
    )

    return sampled_datapoint


def weight_and_normalize_distribution(
        dist: Tensor,
        weights: Tensor
    ) -> Tensor:
    weighted_prob = dist.unsqueeze(-1) * weights        # bs, N, d0, d_t-1
    unnormalized_prob = weighted_prob.sum(dim=-2)       # bs, N, d_t-1
    unnormalized_prob[torch.sum(unnormalized_prob, dim=-1) == 0] = 1e-5
    return unnormalized_prob / torch.sum(unnormalized_prob, dim=-1, keepdim=True) # bs, n, d_t-1
    

def apply_posterior_noise_graph(
        original_datapoint: Tuple[DenseGraph, DenseEdges],
        current_datapoint: Tuple[DenseGraph, DenseEdges],
        noise_graph_bar_t: DenseGraph,
        noise_graph_bar_t_1: DenseGraph,
        noise_graph_t: DenseGraph,
        sigma_sq_ratio, 
        z_t_prefactor, 
        x_prefactor, 
        prefactor1, 
        prefactor2
    ) -> Tuple[DenseGraph, DenseEdges]:

    with_external = current_datapoint[1] is not None

    bs, nq, _, de = current_datapoint[0].edge_adjmat.shape
    
    mu = z_t_prefactor.unsqueeze(-1) * current_datapoint[0].attribute_edge + x_prefactor.unsqueeze(-1) * original_datapoint[0].attribute_edge

    sampled_pos = torch.randn(current_datapoint[0].attribute_edge.shape, device=current_datapoint[0].attribute_edge.device)
    sampled_pos = torch.mul(sampled_pos, torch.triu(current_datapoint[0].edge_mask, diagonal=1).float())
    sampled_pos = sampled_pos + sampled_pos.transpose(1,2)

    sigma2_t_s = prefactor1 - prefactor2
    noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
    noise_prefactor = torch.sqrt(noise_prefactor_sq).unsqueeze(-1)
    
    dist = mu + noise_prefactor.unsqueeze(-1) * sampled_pos  
        
    # compute weights for the parameterization
    p_s_and_t_given_0_X = compute_prob_s_t_given_0(
        X_t =   current_datapoint[0].x,
        Qt =    noise_graph_t.x,
        Qsb =   noise_graph_bar_t_1.x,
        Qtb =   noise_graph_bar_t.x
    )

    p_s_and_t_given_0_E = compute_prob_s_t_given_0(
        X_t =   current_datapoint[0].edge_adjmat,
        Qt =    noise_graph_t.edge_adjmat,
        Qsb =   noise_graph_bar_t_1.edge_adjmat,
        Qtb =   noise_graph_bar_t.edge_adjmat
    )
    
    p_s_and_t_given_0_C = compute_prob_s_t_given_0(
        X_t =   current_datapoint[0].attribute_node,
        Qt =    noise_graph_t.attribute_node,
        Qsb =   noise_graph_bar_t_1.attribute_node,
        Qtb =   noise_graph_bar_t.attribute_node
    )
    
    # in this part i should put the same values compute_prob_s_t_given_0
    # but for current_datapoint[0].attribute_node
    # Then in the part below a section about the denoising of positions values
    
    if with_external:
        p_s_and_t_given_0_E_ext = compute_prob_s_t_given_0(
            X_t =   current_datapoint[1].edge_adjmat,
            Qt =    noise_graph_t.edge_adjmat,
            Qsb =   noise_graph_bar_t_1.edge_adjmat,
            Qtb =   noise_graph_bar_t.edge_adjmat
        )

    # weight the original datapoint probability distribution
    prob_x = weight_and_normalize_distribution(
        dist =      original_datapoint[0].x,
        weights =   p_s_and_t_given_0_X
    )

    prob_e = weight_and_normalize_distribution(
        dist =      original_datapoint[0].edge_adjmat.reshape(bs, -1, de),
        weights =   p_s_and_t_given_0_E
    )
    prob_e = prob_e.reshape(bs, nq, nq, de)
    
    prob_c = weight_and_normalize_distribution(
        dist =      original_datapoint[0].attribute_node,
        weights=    p_s_and_t_given_0_C
    )

    if with_external:
        bs, nq, nk, de = current_datapoint[1].edge_adjmat.shape
        prob_e_ext = weight_and_normalize_distribution(
            dist =      original_datapoint[1].edge_adjmat.reshape(bs, -1, de),
            weights =   p_s_and_t_given_0_E_ext
        )
        prob_e_ext = prob_e_ext.reshape(bs, nq, nk, de)


    prob_graph = DenseGraph(
        x=prob_x,
        edge_adjmat=prob_e,
        y=current_datapoint[0].y,
        attribute_node=prob_c,
        node_mask=current_datapoint[0].node_mask,
        edge_mask=current_datapoint[0].edge_mask,
        attribute_edge = dist
    )

    if with_external:

        prob_ext_edges = DenseEdges(
            edge_adjmat=prob_e_ext,
            edge_mask=current_datapoint[1].edge_mask
        )

    else:
        prob_ext_edges = None

    # create probability graph
    prob_datapoint = (
        prob_graph,
        prob_ext_edges
    )

    # fill out probability graph
    prob_datapoint = fill_out_prob_graph(
        prob_datapoint=prob_datapoint
    )

    # sample from transition probabilities
    sampled_datapoint = sample_from_probabilities(
        prob_datapoint=prob_datapoint
    )

    return sampled_datapoint



def get_num_classes(graph: DenseGraph) -> Tuple[int, int]:
    """
    Parameters
    ----------
    graph : DenseGraph
        the graph to get the number of classes for
    """
    # get number of classes
    x_classes = graph.x.shape[-1]
    e_classes = graph.edge_adjmat.shape[-1]
    c_classes = graph.attribute_node.shape[-1]

    return x_classes, e_classes, c_classes

class DiscreteUniformDiffusionProcess_distance(NoiseProcess):

    def __init__(
            self,
            schedule : NoiseSchedule
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """
        # call super for the NoiseProcess
        super().__init__(schedule=schedule)

        """
        # setup number of classes
        self.x_classes = x_classes
        self.e_classes = e_classes

        # pre-build uniform transition probabilities
        self.u_x = torch.ones(1, self.x_classes, self.x_classes)
        if self.x_classes > 0:
            self.u_x = self.u_x / self.x_classes

        self.u_e = torch.ones(1, self.e_classes, self.e_classes)
        if self.e_classes > 0:
            self.u_e = self.u_e / self.e_classes
        """

    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            num_new_nodes: IntTensor,
            ext_node_mask: BoolTensor,
            num_classes: Dict[str, int]
        ) -> Tuple[DenseGraph, DenseEdges]:
        # num new nodes has shape (bs,), and each element
        # is the number of nodes the graph should have
        bs = len(num_new_nodes)
        max_num_nodes = num_new_nodes.max().item()

        # get number of classes
        x_classes, e_classes, c_classes = num_classes[DIM_X], num_classes[DIM_E], num_classes[DIM_C]


        # get current device
        device = num_new_nodes.device

        # in this case the sampling strategy is done by considering uniform distribution and not 
        # the training marginal one
        # ADD TRAINING MARGINAL DISTRIBUTION

        x = torch.randint(low=0, high=x_classes, size=(bs, max_num_nodes), device=device)
        node_mask = torch.arange(max_num_nodes, device=device) < num_new_nodes.unsqueeze(-1)

        # generate uniform edge adjmat, without self loops
        edge_adjmat = torch.randint(low=0, high=e_classes, size=(bs, max_num_nodes, max_num_nodes), device=device)
        edge_adjmat = torch.triu(edge_adjmat, diagonal=1)
        edge_adjmat = edge_adjmat + edge_adjmat.transpose(1, 2)
        edge_mask = get_edge_mask_dense(node_mask)
        
        # generate the attribute node values
        
        c = torch.randint(low=0, high=c_classes, size=(bs, max_num_nodes), device=device)
        
        # generate the position values from standard normal distribution
        

        # this part doesn't make sense to me, it should be not symmetric but in the
        # end it's symmetric, I dont kown !!!
        dist = torch.randn(size=(bs, max_num_nodes, max_num_nodes), device=device)
        dist = torch.mul(dist, torch.triu(edge_mask, diagonal=1).float())
        dist = dist + dist.transpose(1,2)
        #pos = remove_mean_with_mask(pos, node_mask)

        # compose graph
        graph = DenseGraph(
            x =				x,
            edge_adjmat =	edge_adjmat,
            y =				None,
            attribute_node=         c,
            attribute_edge=       dist,
            node_mask =		node_mask,
            edge_mask =     edge_mask
        ).apply_mask()


        ext_edges = None
        if ext_node_mask is not None:
            # generate uniform external edge adjmat
            max_ext_nodes = ext_node_mask.shape[1]
            ext_edge_adjmat = torch.randint(low=0, high=e_classes, size=(bs, max_num_nodes, max_ext_nodes), device=device)
            # mask out fake nodes
            ext_edge_mask = get_bipartite_edge_mask_dense(node_mask, ext_node_mask)
            
            ext_edges = DenseEdges(
                edge_adjmat =   ext_edge_adjmat,
                edge_mask =     ext_edge_mask
            ).apply_mask()

        return graph, ext_edges


    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(self, current_datapoint: Tuple[DenseGraph, DenseEdges], t: IntTensor, **kwargs):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K
        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de).
        """

        # get diffusion parameter
        beta_t: Tensor = self.get_params_next(t, **kwargs).unsqueeze(-1).unsqueeze(-1)

        # get current device
        graph, ext_edges = current_datapoint
        device = graph.device

        # get number of classes
        x_classes, e_classes, c_classes = get_num_classes(graph)

        # exact definition from the discrete diffusion paper
        q_x = (1 - beta_t) * torch.eye(x_classes, device=device).unsqueeze(0) + beta_t / x_classes
        q_e = (1 - beta_t) * torch.eye(e_classes, device=device).unsqueeze(0) + beta_t / e_classes
        q_c = (1 - beta_t) * torch.eye(c_classes, device=device).unsqueeze(0) + beta_t / c_classes
        
        # add noise inside positions
        
        noise_dist = torch.randn(graph.attribute_edge.shape, device=device)
        noise_pos_masked = torch.mul(noise_dist, torch.triu(graph.edge_mask, diagonal=1).float())
        noise_pos_masked = noise_pos_masked + noise_pos_masked.transpose(1,2)
        
        # this should be alpha and (1-alpha)
        s=self.get_sigma_bar(t=t).unsqueeze(-1).unsqueeze(-1)
        
        dist_t = (1-beta_t) * graph.attribute_edge + s*noise_pos_masked

        transition_graph = DenseGraph(
            x=q_x,
            edge_adjmat=q_e,
            y=None,
            attribute_node=q_c,
            attribute_edge=dist_t
        )

        return transition_graph


    def apply_noise_next(
            self,
            current_datapoint: Tuple[DenseGraph, DenseEdges],
            noise: DenseGraph,
            t: IntTensor,
            **kwargs
        ) -> Tuple[DenseGraph, DenseEdges]:

        graph, ext_edges = apply_noise_graph(
            datapoint =			current_datapoint,
            noise_graph =		noise
        )

        return graph, ext_edges

    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################

    def sample_noise_from_original(
            self,
            original_datapoint: DenseGraph|Tuple[DenseGraph, DenseEdges],
            t: IntTensor,
            **kwargs
        ):
        
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K
        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de).
        """

        # get diffusion parameter
        alpha_bar_t: Tensor = self.get_params_from_original(t, **kwargs).unsqueeze(-1).unsqueeze(-1)

        # get current device
        graph, ext_edges = original_datapoint
        device = graph.device

        # get number of classes
        x_classes, e_classes, c_classes = get_num_classes(graph)


        # exact definition from the discrete diffusion paper
        q_x = alpha_bar_t * torch.eye(x_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) / x_classes
        q_e = alpha_bar_t * torch.eye(e_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) / e_classes
        q_c = alpha_bar_t * torch.eye(c_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) / c_classes

        # add noise inside positions
        triang_edge_mask = torch.tril(original_datapoint[0].edge_mask, diagonal=-1)
        noise_dist = torch.randn(original_datapoint[0].attribute_edge.shape, device=device)
        noise_dist_masked = torch.mul(noise_dist.squeeze(-1), triang_edge_mask.float())
        noise_dist_masked = (noise_dist_masked + torch.transpose(noise_dist_masked, 1, 2))
        
        # this should be alpha and (1-alpha)
        dist_t = (alpha_bar_t) * original_datapoint[0].attribute_edge + (self.get_sigma_bar(t=t).unsqueeze(-1).unsqueeze(-1))*noise_dist_masked

        transition_graph = DenseGraph(
            x=q_x,
            edge_adjmat=q_e,
            y=None,
            attribute_node=q_c,
            attribute_edge=dist_t
        )

        return transition_graph


    def apply_noise_from_original(
            self,
            original_datapoint: Tuple[DenseGraph, DenseEdges],
            noise: torch.BoolTensor,
            t: IntTensor,
            **kwargs
        ) -> Tuple[DenseGraph, DenseEdges]:


        graph, ext_edges = apply_noise_graph(
            datapoint =			original_datapoint,
            noise_graph =		noise
        )

        return graph, ext_edges
    
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(
            self,
            original_datapoint: Tuple[DenseGraph, DenseEdges],
            current_datapoint: Tuple[DenseGraph, DenseEdges],
            t: IntTensor,
            **kwargs
        ) -> Tuple[DenseGraph, DenseGraph, DenseGraph]:

        trans_graph_bar_t = self.sample_noise_from_original(original_datapoint, t, **kwargs)
        trans_graph_bar_t_minus_one = self.sample_noise_from_original(original_datapoint, t-1, **kwargs)
        trans_graph_t = self.sample_noise_next(current_datapoint, t, **kwargs)
        
        sigma_sq_ratio = self.get_sigma_pos_sq_ratio(t-1, t)
        z_t_prefactor = (self.get_alpha_pos_ts(t=t, s=t-1) * sigma_sq_ratio).unsqueeze(-1)
        x_prefactor = self.get_x_pos_prefactor(s=t-1, t=t).unsqueeze(-1)
        
        prefactor1 = self.get_sigma_bar(t=t)
        prefactor2 = self.get_sigma_bar(t=t-1) * self.get_alpha_pos_ts_sq(t=t, s=t-1)
        
        return (
            trans_graph_bar_t,
            trans_graph_bar_t_minus_one,
            trans_graph_t,
            sigma_sq_ratio,
            z_t_prefactor,
            x_prefactor,
            prefactor1,
            prefactor2
        )


    def apply_noise_posterior(
            self,
            original_datapoint: Tuple[DenseGraph, DenseEdges],
            current_datapoint: Tuple[DenseGraph, DenseEdges],
            noise: Tuple[DenseGraph, DenseGraph, DenseGraph],
            t: IntTensor,
            **kwargs
        ) -> Tuple[DenseGraph, DenseEdges]:

        noise_graph_bar_t, noise_graph_bar_t_minus_one, noise_graph_t, sigma_sq_ratio, z_t_prefactor, x_prefactor, prefactor1, prefactor2 = noise

        graph, ext_edges = apply_posterior_noise_graph(
            original_datapoint =	original_datapoint,
            current_datapoint =		current_datapoint,
            noise_graph_bar_t =		noise_graph_bar_t,
            noise_graph_bar_t_1 =	noise_graph_bar_t_minus_one,
            noise_graph_t =			noise_graph_t,
            sigma_sq_ratio = sigma_sq_ratio,
            z_t_prefactor =z_t_prefactor,
            x_prefactor = x_prefactor,
            prefactor1 = prefactor1,
            prefactor2 = prefactor2,
        )

        return graph, ext_edges


################################################################################
#                            RESOLVE OBJECT BY NAME                            #
################################################################################

DIFFUSION_SCHEDULE_COSINE = 'cosine_distance'

DIFFUSION_PROCESS_DISCRETE = 'discrete_uniform_distance'

def resolve_graph_diffusion_schedule(name: str) -> type:
    if name == DIFFUSION_SCHEDULE_COSINE:
        return CosineDiffusionSchedule_distance

    else:
        raise DiffusionProcessException(f'Could not resolve diffusion schedule name: {name}')

def resolve_graph_diffusion_process(name: str) -> type:
    if name == DIFFUSION_PROCESS_DISCRETE:
        return DiscreteUniformDiffusionProcess_distance
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion process name: {name}')
    
