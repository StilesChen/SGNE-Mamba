import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.ER_edge_encoder import EREdgeEncoder
import logging

from graphgps.layer.gps_layer import GPSLayer
from graphgps.transform.structural_dfs import (prepare_structural_dfs, structural_dfs_seed,
                                              activate_structural_dfs)


def remove_unused_terminal_edge_affine(layer):
    """Remove affine parameters on the unused final node-only edge output.

    GatedGCN computes node messages BEFORE this edge BatchNorm. Its running
    statistics and normalization stay intact, but its affine parameters cannot
    affect a node-only graph readout and must not inflate the capacity budget.
    """
    norm = layer.local_model.bn_edge_e
    norm.affine = False
    norm.register_parameter('weight', None)
    norm.register_parameter('bias', None)


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-set edge dim for PNA.
            cfg.gnn.dim_edge = 16 if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            if cfg.dataset.edge_encoder_name == 'ER':
                self.edge_encoder = EREdgeEncoder(cfg.gnn.dim_edge)
            elif cfg.dataset.edge_encoder_name.endswith('+ER'):
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name[:-3]]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge - cfg.posenc_ERE.dim_pe)
                self.edge_encoder_er = EREdgeEncoder(cfg.posenc_ERE.dim_pe, use_edge_attr=True)
            else:
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)

            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                    has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            # Bare Mamba_Edge preserves the project's usual local GatedGCN.
            local_gnn_type, global_model_type = (
                ('CustomGatedGCN', 'Mamba_Edge') if cfg.gt.layer_type == 'Mamba_Edge'
                else cfg.gt.layer_type.split('+'))
        except ValueError:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        self.structural_edge_scan = (global_model_type == 'Mamba_Edge'
                                     and getattr(cfg.gt, 'scan_target', 'node') != 'node')
        self.edge_state_transfer = getattr(cfg.gt, 'edge_state_transfer', 'full')
        if self.edge_state_transfer not in ('full', 'local_only', 'reset_initial'):
            raise ValueError(f'Unsupported edge_state_transfer: {self.edge_state_transfer}')
        if self.edge_state_transfer != 'full' and not (
                local_gnn_type == 'CustomGatedGCN' and self.structural_edge_scan
                and getattr(cfg.gt, 'scan_target', 'node') == 'both'):
            raise ValueError('Edge-state ablations require CustomGatedGCN+Mamba_Edge and scan_target=both')
        if self.edge_state_transfer == 'reset_initial' and cfg.gnn.layers_pre_mp != 0:
            raise ValueError('reset_initial requires layers_pre_mp=0 to define the initial encoded edge state')
        if self.structural_edge_scan:
            self.structural_scan_order = getattr(cfg.gt, 'edge_scan_order', 'node_edge_dfs').replace('node_edge_', '', 1)
            self.structural_segmentation = getattr(cfg.gt, 'edge_scan_segmentation', 'native')
            self.structural_max_len = int(getattr(cfg.gt, 'edge_scan_max_segment_len', 512))
            self.structural_shuffle = getattr(cfg.gt, 'edge_scan_within_segment_shuffle', False)
            self.structural_use_attributes = getattr(cfg.gt, 'edge_dfs_use_attributes', True)
            if self.structural_segmentation not in ('native', 'fixed_chunks') or self.structural_max_len < 1:
                raise ValueError('Invalid structural segmentation or maximum segment length')
            self.structural_dfs_rounds = int(getattr(cfg.gt, 'edge_dfs_signature_rounds', 2))
            if self.structural_dfs_rounds < 0:
                raise ValueError('edge_dfs_signature_rounds must be non-negative')
            self.register_buffer('_structural_dfs_step', torch.zeros((), dtype=torch.long))
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                enable_reverse_mamba=cfg.gt.enable_reverse_mamba,
                fusion_mode=cfg.gt.fusion_mode,
                fixed_weight=cfg.gt.fixed_weight,
                scan_target=getattr(cfg.gt, 'scan_target', 'node'),
                edge_dim=getattr(cfg.gnn, 'dim_edge', cfg.gt.dim_hidden),
                edge_scan_order=getattr(cfg.gt, 'edge_scan_order', 'dfs'),
                node_edge_scale_mode=getattr(cfg.gt, 'node_edge_scale_mode', 'dim'),
                node_edge_scale_value=getattr(cfg.gt, 'node_edge_scale_value', 1.0),
                global_scale_mode=getattr(cfg.gt, 'global_scale_mode', 'none'),
                global_scale_value=getattr(cfg.gt, 'global_scale_value', 1.0),
                mamba_tokenization_m=getattr(cfg.gt, 'mamba_tokenization_m', 0),
                edge_node_fusion_mode=getattr(cfg.gt, 'edge_node_fusion_mode', 'fixed'),
                edge_node_weight=getattr(cfg.gt, 'edge_node_weight', 0.5),
                numerical_debug=getattr(cfg.train, 'numerical_debug', False),
                numerical_debug_threshold=getattr(cfg.train, 'numerical_debug_threshold', 1e12),
                edge_mamba_residual_scale=getattr(cfg.gt, 'edge_mamba_residual_scale', 0.1),
                edge_input_projection_optimized=getattr(cfg.gt, 'edge_input_projection_optimized', True),
                topology_cache=getattr(cfg.gt, 'topology_cache', True),
                edge_state_transfer=self.edge_state_transfer,
            ))
        self.layers = torch.nn.Sequential(*layers)

        if (global_model_type == 'Mamba_Edge'
                and getattr(cfg.gt, 'scan_target', 'node') == 'node'
                and local_gnn_type == 'CustomGatedGCN'
                and cfg.dataset.task == 'graph'
                and cfg.gnn.head in ('default', 'graph') and layers):
            remove_unused_terminal_edge_affine(layers[-1])

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def prepare_structural_batch(self, batch, *, seed=None):
        if self.structural_edge_scan:
            if seed is None:
                seed = structural_dfs_seed(torch.initial_seed(), self.training,
                                           self._structural_dfs_step.item())
            prepare_structural_dfs(batch, self.structural_dfs_rounds,
                                   torch.Generator().manual_seed(seed),
                                   scan_order=self.structural_scan_order,
                                   segmentation=self.structural_segmentation,
                                   max_path_len=self.structural_max_len,
                                   within_segment_shuffle=self.structural_shuffle,
                                   use_attributes=self.structural_use_attributes,
                                   shuffle_generator=torch.Generator().manual_seed(
                                       (seed + 2000003) % (2**63 - 1)))
            # Public tensors participate in PyG's device transfer; consume them
            # in forward before heads reconstruct individual graphs.
            for name, value in zip(('edges', 'batch', 'order', 'paths'), batch._gps_structural_dfs):
                setattr(batch, 'dfs_runtime_' + name, value)
            batch._gps_structural_prepared = True

    def forward(self, batch):
        # Counts are shared by layers of this forward, never hidden features.
        if hasattr(batch, '_gps_topology_cache'):
            delattr(batch, '_gps_topology_cache')
        if self.structural_edge_scan:
            if not getattr(batch, '_gps_structural_prepared', False):
                self.prepare_structural_batch(batch)
            activate_structural_dfs(batch)
            if self.training:
                self._structural_dfs_step.add_(1)
        # 保证在 encoder 运行前完成类型转换
        if cfg.dataset.edge_encoder and getattr(cfg.dataset, 'edge_encoder_name', '') == 'Bond':
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                # 保持 device；只在确认为整数索引时转换
                # 推荐使用 to(torch.long) 而不是 long()，以便保留 device
                batch.edge_attr = batch.edge_attr.to(torch.long).to(
                    batch.x.device if hasattr(batch, 'x') and batch.x is not None else batch.edge_attr.device)

        initial_edge = None
        for i, module in enumerate(self.children()):
            if module is self.layers and self.edge_state_transfer == 'reset_initial':
                if initial_edge is None:
                    raise ValueError('reset_initial requires encoded edge_attr')
                for layer in self.layers:
                    # Reset before the layer, never before its scan/fusion.
                    # Keep the autograd connection to the shared input encoder.
                    batch.edge_attr = initial_edge
                    batch = layer(batch)
            else:
                batch = module(batch)

            # Keep recovery at the same module boundary, but transfer the
            # feature checks together (one host synchronization per device).
            checks_by_device = {}
            for name in ('x', 'edge_attr'):
                value = getattr(batch, name, None)
                if value is not None:
                    checks_by_device.setdefault(value.device, []).append(
                        (name, torch.isfinite(value).all()))
            finite = {}
            for checks in checks_by_device.values():
                flags = torch.stack([flag for _, flag in checks]).tolist()
                finite.update((name, flag) for (name, _), flag in zip(checks, flags))

            if hasattr(batch, 'x') and batch.x is not None:
                if not finite['x']:
                    num_bad = (~torch.isfinite(batch.x)).sum().item()
                    logging.warning(
                        f"Non-finite values detected after module {i} ({module.__class__.__name__}): "
                        f"{num_bad}/{batch.x.numel()} elements. Applying recovery..."
                    )
                    batch.x = torch.nan_to_num(batch.x, nan=0.0, posinf=1e4, neginf=-1e4)

            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                if not finite['edge_attr']:
                    num_bad = (~torch.isfinite(batch.edge_attr)).sum().item()
                    logging.warning(
                        f"Non-finite edge_attr after module {i}: {num_bad}/{batch.edge_attr.numel()} elements. "
                        f"Applying recovery..."
                    )
                    batch.edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)

            if module is self.encoder and self.edge_state_transfer == 'reset_initial':
                initial_edge = batch.edge_attr

        return batch
