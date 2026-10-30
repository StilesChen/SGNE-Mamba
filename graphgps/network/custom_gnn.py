import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer


@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        # AtomEncoder uses nn.Embedding, so its input must contain integer
        # category indices. Some preprocessing paths may promote x to float.
        # Cast only before the encoder (never after it), when the configured
        # encoder interprets x as categorical atom features.
        node_encoder_name = getattr(cfg.dataset, 'node_encoder_name', '')
        if (cfg.dataset.node_encoder and node_encoder_name.startswith('Atom')
                and hasattr(batch, 'x') and batch.x is not None):
            batch.x = batch.x.long()

        # BondEncoder also uses nn.Embedding for categorical bond features.
        edge_encoder_name = getattr(cfg.dataset, 'edge_encoder_name', '')
        if (cfg.dataset.edge_encoder and edge_encoder_name == 'Bond'
                and hasattr(batch, 'edge_attr') and batch.edge_attr is not None):
            batch.edge_attr = batch.edge_attr.long()

        for module in self.children():
            batch = module(batch)
        return batch
