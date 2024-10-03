r"""
The Graph Neural Network from the `"How Powerful are Graph Neural Networks?"
<https://arxiv.org/abs/1810.00826>`_ paper.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor
from torch.nn import Identity

from BaseGNN import GNNBasic, BasicEncoder


import numpy as np
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import json
import scipy.sparse as sp


def to_scipy_sparse_matrix(adj_matrix):
    # Assuming adj_matrix is a PyTorch tensor
    return sp.csr_matrix(adj_matrix.cpu().numpy())


# def from_scipy_sparse_matrix(sparse_matrix):
#     # Create edge index from scipy sparse matrix
#     return torch.tensor([sparse_matrix.row, sparse_matrix.col], dtype=torch.long)


def create_data_objects(xs, adjs, answers):
    # Iterate over each batch
    data_list = []
    for batch_idx in tqdm(range(len(xs)), desc="Graph Batch", leave=False):
        x = xs[batch_idx]  # Node features for the batch
        adj_batch = adjs[batch_idx]  # Adjacency matrix for the batch
        edge_indexes = [from_scipy_sparse_matrix(to_scipy_sparse_matrix(adj))[0] for adj in adj_batch]
        for graph_idx in range(x.size(0)):  # Iterate through each graph in the batch
            node_features = x[graph_idx]  # Node features for the graph
            edge_idx = edge_indexes[graph_idx]  # Edge index for the graph
            answer = answers[batch_idx * x.size(0) + graph_idx]  # answers[graph_idx] Label for the graph
            # Create a Data object
            data = Data(x=node_features, edge_index=edge_idx, y=answer)
            data_list.append(data.cuda())

    # Combine all the data into a single batch (if needed)
    batch_data = Batch.from_data_list(data_list)
    return batch_data


def create_data_objects_batch(xs, adj_batch, answers=None):
    data_list = []
    edge_indexes = [from_scipy_sparse_matrix(to_scipy_sparse_matrix(adj))[0] for adj in adj_batch]
    for graph_idx in range(xs.size(0)):  # Iterate through each graph in the batch
        node_features = xs[graph_idx]  # Node features for the graph
        edge_idx = edge_indexes[graph_idx] # Edge index for the graph
        if answers is not None:
            answer = answers[graph_idx]  # Label for the graph
        else:
            answer = torch.tensor([0], dtype=torch.float32)
        # Create a Data object
        data = Data(x=node_features, edge_index=edge_idx, y=answer)
        data_list.append(data.cuda())
    # Combine all the data into a single batch (if needed)
    batch_data = Batch.from_data_list(data_list)
    return batch_data


def create_data_objects_eval(xs, adjs, answers):
    data_list = []
    for i, (x_batch, adj_batch) in enumerate(zip(xs, adjs)):
        edge_indexes = [from_scipy_sparse_matrix(to_scipy_sparse_matrix(adj))[0] for adj in adj_batch]
        for j in range(x_batch.size(0)):  # Assuming x_batch is shaped [batch_size, num_nodes, num_features]
            data = Data(x=x_batch[j], edge_index=edge_indexes[j], y=answers[i * x_batch.size(0) + j])   # answers[j]
            data_list.append(data.cuda())
    return data_list


def nan2zero_get_mask(data):
    mask = ~torch.isnan(data.y)
    targets = torch.clone(data.y).detach()
    targets[~mask] = 0

    return mask, targets


class BaseModel(nn.Module):
    def __init__(self, config, x_mid_v, adj_mid_v, val_targets):
        super().__init__()
        self.GNNmodel = GIN(config)
        self.loss_fct = nn.MSELoss()
        self.val_data_list = create_data_objects_eval(x_mid_v, adj_mid_v, val_targets)

    def forward(self, x_mid, adj_mid, targets):
        r"""
        The GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        train_data =create_data_objects(x_mid, adj_mid, targets)
        mask, labels = nan2zero_get_mask(train_data)  #nan2zero_get_mask is a function that returns a mask and targets
        edge_weight = None  # edge_weight is not usually needed for graph-level tasks
        out = self.GNNmodel(data=train_data, edge_weight=edge_weight, targets=labels, mask=mask)
        loss = self.loss_fct(out.view(-1), labels.view(-1))
        return loss

    @torch.no_grad()
    def pred_out(self, x_batch, adj_batch, targets=None):
        train_data =create_data_objects_batch(x_batch, adj_batch, targets)
        mask, labels = nan2zero_get_mask(train_data)  #nan2zero_get_mask is a function that returns a mask and targets
        edge_weight = None  # edge_weight is not usually needed for graph-level tasks
        out = self.GNNmodel(data=train_data, edge_weight=edge_weight, targets=labels, mask=mask)
        return out

    @torch.no_grad()
    def evaluate_seq_step(self, bs):
        losses = []
        loader = DataLoader(self.val_data_list, batch_size=bs, shuffle=False)
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Validation")
        for index, data in pbar:
            mask, targets = nan2zero_get_mask(data)  # nan2zero_get_mask is a function that returns a mask and targets
            edge_weight = None  # edge_weight is not usually needed for graph-level tasks
            out = self.GNNmodel(data=data, edge_weight=edge_weight, targets=targets, mask=mask)
            loss = self.loss_fct(out.view(-1), targets.view(-1))
            losses.append(loss.item())
        return losses

    def configure_optimizers(self, train_config, logger):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # if 'ref_model' not in fpn and 'reward_model' not in fpn:
                #     no_decay.add(fpn)
                decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                             % (str(param_dict.keys() - union_params), )
        if len(param_dict.keys() - union_params) != 0:
            logger.log(f"skipping param: {param_dict.keys() - union_params}")

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], 'lr': train_config.learning_rate * 2,
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=(0.9, 0.95))
        return optimizer


class GIN(GNNBasic):
    r"""
    The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`)
    """

    def __init__(self, config):

        super().__init__(config)
        self.feat_encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)
        out = self.classifier(out_readout)
        return out


class GINFeatExtractor(GNNBasic):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`)
    """
    def __init__(self, config, **kwargs):
        super(GINFeatExtractor, self).__init__(config)
        # num_layer = config.model.model_layer
        # if config.dataset.dataset_type == 'mol':
        #     self.encoder = GINMolEncoder(config, **kwargs)
        #     self.edge_feat = True
        # else:
        self.encoder = GINEncoder(config, **kwargs)
        self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        return out_readout


class GINEncoder(BasicEncoder):
    r"""
    The GIN encoder for non-molecule data, using the :class:`~GINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config, *args, **kwargs):

        super(GINEncoder, self).__init__(config, *args, **kwargs)
        num_layer = config.model.model_layer
        self.without_readout = kwargs.get('without_readout')

        # self.atom_encoder = AtomEncoder(config.model.dim_hidden)

        if kwargs.get('without_embed'):
            self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
        else:
            self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.dataset.dim_node, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

        self.convs = nn.ModuleList(
            [
                gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            node feature representations
        """

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout


class Classifier(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config):

        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(*(
        #         [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_ffn), nn.BatchNorm1d(2 * config.model.dim_ffn)] +
        #         [nn.ReLU(), nn.Linear(2 * config.model.dim_ffn, config.dataset.num_classes)]
        # ))
        self.classifier = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden, 1)]   # config.dataset.num_classes
        ))

    def forward(self, feat: Tensor) -> Tensor:
        r"""
        Applies a linear transformation to feature representations.

        Args:
            feat (Tensor): feature representations

        Returns (Tensor):
            label predictions

        """
        return self.classifier(feat)
