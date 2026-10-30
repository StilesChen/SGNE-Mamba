from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import logging
import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import degree, to_dense_batch

from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.transform.dfs_scan import dfs_edge_paths, assemble_dfs_scan_paths
from mamba_ssm import Mamba

# Keep old behavior for reproducibility.
torch.manual_seed(0)


def permute_nodes_within_identity(identities: torch.Tensor) -> torch.Tensor:
    unique_identities = torch.unique(identities)
    node_indices = torch.arange(len(identities), device=identities.device)
    masks = identities.unsqueeze(0) == unique_identities.unsqueeze(1)
    permuted_indices = torch.cat(
        [node_indices[mask][torch.randperm(int(mask.sum()), device=identities.device)] for mask in masks]
    )
    return permuted_indices


def sort_rand_gpu(pop_size: int, num_samples: int, neighbours: torch.Tensor) -> torch.Tensor:
    idx_select = torch.argsort(torch.rand(pop_size, device=neighbours.device))[:num_samples]
    return neighbours[idx_select]


def augment_seq(edge_index: torch.Tensor, batch: torch.Tensor, num_k: int = -1):
    unique_batches = torch.unique(batch)
    permuted_indices = []
    mask = []

    for batch_index in unique_batches:
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        for k in indices_in_batch:
            neighbours = edge_index[1][edge_index[0] == k]
            if num_k > 0 and len(neighbours) > num_k:
                neighbours = sort_rand_gpu(len(neighbours), num_k, neighbours)
            permuted_indices.append(neighbours)
            mask.append(torch.zeros(neighbours.shape, dtype=torch.bool, device=batch.device))
            permuted_indices.append(torch.tensor([k], device=batch.device))
            mask.append(torch.tensor([1], dtype=torch.bool, device=batch.device))

    permuted_indices = torch.cat(permuted_indices)
    mask = torch.cat(mask)
    return permuted_indices.to(device=batch.device), mask.to(device=batch.device)


def lexsort(keys: List[torch.Tensor], dim: int = -1, descending: bool = False) -> torch.Tensor:
    """Stable indirect sort using multiple keys (last key has highest priority)."""
    assert len(keys) >= 1
    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out


def permute_within_batch(batch: torch.Tensor) -> torch.Tensor:
    unique_batches = torch.unique(batch)
    permuted_indices = []
    for batch_index in unique_batches:
        indices_in_batch = (batch == batch_index).nonzero().view(-1)
        permuted_indices_in_batch = indices_in_batch[
            torch.randperm(indices_in_batch.numel(), device=batch.device)
        ]
        permuted_indices.append(permuted_indices_in_batch)
    return torch.cat(permuted_indices)


def scatter_mean_fallback(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """scatter_mean over dim=0 with a pure-torch fallback (avoids hard dep on torch_scatter)."""
    try:
        from torch_scatter import scatter_mean  # type: ignore

        return scatter_mean(src, index, dim=0, dim_size=dim_size)
    except Exception:
        if src.dim() == 1:
            src_ = src.unsqueeze(-1)
        else:
            src_ = src
        out = src_.new_zeros((dim_size, src_.size(-1)))
        cnt = src_.new_zeros((dim_size, 1))
        out.index_add_(0, index, src_)
        ones = torch.ones((src_.size(0), 1), device=src_.device, dtype=src_.dtype)
        cnt.index_add_(0, index, ones)
        out = out / cnt.clamp(min=1)
        return out if src.dim() > 1 else out.squeeze(-1)


class StableLayerNorm(nn.Module):
    """LayerNorm that avoids squaring very large float32 inputs.

    Scaling every row by its maximum absolute value and scaling eps by the same
    factor is algebraically equivalent to LayerNorm, while keeping the variance
    computation in a safe numerical range.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        scaled = x / scale
        mean = scaled.mean(dim=-1, keepdim=True)
        variance = (scaled - mean).square().mean(dim=-1, keepdim=True)
        # Written as two divisions so scale**2 cannot overflow.
        scaled_eps = self.eps / scale / scale
        out = (scaled - mean) * torch.rsqrt(variance + scaled_eps)
        if self.weight is not None:
            out = out * self.weight + self.bias
        return out


class GPSLayer(nn.Module):
    """Local MPNN + global sequence model layer.

    This fork supports Mamba node-scan, edge-scan, and a new parallel mode
    ``scan_target='both'`` that runs node-scan and edge-scan in parallel and
    fuses their node-level outputs.
    """

    def __init__(
            self,
            dim_h,
            local_gnn_type,
            global_model_type,
            num_heads,
            pna_degrees=None,
            equivstable_pe=False,
            dropout=0.0,
            attn_dropout=0.0,
            layer_norm=False,
            batch_norm=True,
            bigbird_cfg=None,
            enable_reverse_mamba: bool = False,
            fusion_mode: str = 'fixed',
            fixed_weight: float = 0.5,
            scan_target: str = 'node',
            edge_dim=None,
            edge_scan_order: str = 'dfs',
            node_edge_scale_mode: str = 'dim',
            node_edge_scale_value: float = 1.0,
            global_scale_mode: str = 'none',
            global_scale_value: float = 1.0,
            mamba_tokenization_m: int = 0,
            edge_node_fusion_mode=None,
            numerical_debug: bool = False,
            numerical_debug_threshold: float = 1e12,
            edge_mamba_residual_scale: float = 0.1,
            edge_node_weight=None,
            edge_input_projection_optimized: bool = True,
            topology_cache: bool = True,
            edge_state_transfer: str = 'full',
    ):
        super().__init__()

        self.dim_h = dim_h
        self.edge_state_transfer = edge_state_transfer
        if edge_state_transfer not in ('full', 'local_only', 'reset_initial'):
            raise ValueError(f'Unsupported edge_state_transfer: {edge_state_transfer}')
        if edge_state_transfer != 'full' and not (
                local_gnn_type == 'CustomGatedGCN'
                and global_model_type == 'Mamba_Edge' and scan_target == 'both'):
            raise ValueError('Edge-state ablations require CustomGatedGCN+Mamba_Edge and scan_target=both')
        self.num_heads = num_heads
        self.edge_input_projection_optimized = bool(edge_input_projection_optimized)
        self.topology_cache = bool(topology_cache)
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.NUM_BUCKETS = 3
        self.numerical_debug = bool(numerical_debug)
        self.numerical_debug_threshold = float(numerical_debug_threshold)
        self.edge_mamba_residual_scale = float(edge_mamba_residual_scale)
        if self.edge_mamba_residual_scale < 0:
            raise ValueError("edge_mamba_residual_scale must be non-negative")

        self.enable_reverse_mamba = enable_reverse_mamba
        self.fusion_mode = fusion_mode
        self.fixed_weight = fixed_weight
        self.concat_proj = None
        self.gate_layer = None

        self.scan_target = scan_target
        if self.scan_target not in ['node', 'edge', 'both']:
            raise ValueError(f"Unsupported scan_target: {self.scan_target}")

        self.edge_scan_order = edge_scan_order
        if self.edge_scan_order not in ['default', 'dfs', 'node_edge_dfs', 'random',
                                        'node_edge_bfs', 'node_edge_random']:
            raise ValueError(f"Unsupported edge_scan_order: {self.edge_scan_order}")

        if edge_scan_order in ('node_edge_bfs', 'node_edge_random') and global_model_type != 'Mamba_Edge':
            raise ValueError('node_edge_bfs/node_edge_random require Mamba_Edge')
        self.node_edge_scan = edge_scan_order in ('node_edge_dfs', 'node_edge_bfs', 'node_edge_random')

        self.node_edge_scale_mode = node_edge_scale_mode
        if self.node_edge_scale_mode not in ['none', 'dim', 'fixed', 'learnable']:
            raise ValueError(f"Unsupported node_edge_scale_mode: {self.node_edge_scale_mode}")
        self.node_edge_scale_value = float(node_edge_scale_value)
        self.node_edge_scale_param = None
        if self.node_edge_scale_mode == 'learnable':
            self.node_edge_scale_param = nn.Parameter(torch.tensor(self.node_edge_scale_value, dtype=torch.float32))

        self.global_scale_mode = global_scale_mode
        if self.global_scale_mode not in ['none', 'dim', 'fixed', 'learnable']:
            raise ValueError(f"Unsupported global_scale_mode: {self.global_scale_mode}")
        self.global_scale_value = float(global_scale_value)
        self.global_scale_param = None
        if self.global_scale_mode == 'learnable':
            self.global_scale_param = nn.Parameter(torch.tensor(self.global_scale_value, dtype=torch.float32))

        # Edge scan input projection.
        if scan_target in ['edge', 'both']:
            if edge_dim is None:
                edge_dim = dim_h
            if self.node_edge_scan:
                edge_dim = edge_dim + 2 * dim_h
            self.edge_scan_input_proj = nn.Identity() if edge_dim == dim_h else nn.Linear(edge_dim, dim_h)
        else:
            self.edge_scan_input_proj = None

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), nn.ReLU(), Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h,
            )
        elif local_gnn_type == 'PNA':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(
                dim_h,
                dim_h,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=16,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(
                dim_h,
                dim_h,
                dropout=dropout,
                residual=True,
                equivstable_pe=equivstable_pe,
            )
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global sequence model.
        global_model_type_lower = global_model_type.lower()
        if global_model_type == 'None' or (global_model_type == 'Mamba_Edge' and scan_target == 'edge'):
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(dim=dim_h, heads=num_heads, dropout=self.attn_dropout, causal=False)
        elif global_model_type == 'BigBird':
            if bigbird_cfg is None:
                raise ValueError("bigbird_cfg must be provided when global_model_type='BigBird'")
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        elif 'mamba' in global_model_type_lower:
            # Keep the original behavior for the many Mamba_* variants.
            suffix = global_model_type.split('_')[-1]
            if suffix == '2':
                self.self_attn = Mamba(d_model=dim_h, d_state=8, d_conv=4, expand=2)
            elif suffix == '4':
                self.self_attn = Mamba(d_model=dim_h, d_state=4, d_conv=4, expand=4)
            elif suffix == 'SmallConv':
                self.self_attn = Mamba(d_model=dim_h, d_state=16, d_conv=2, expand=1)
            elif suffix == 'SmallState':
                self.self_attn = Mamba(d_model=dim_h, d_state=8, d_conv=4, expand=1)
            else:
                self.self_attn = Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)
        else:
            raise ValueError(f"Unsupported global x-former model: {global_model_type}")
        self.global_model_type = global_model_type
        self.mamba_tokenization_m = int(mamba_tokenization_m)
        if self.mamba_tokenization_m < 0:
            raise ValueError("mamba_tokenization_m must be >= 0")

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

        # Reverse Mamba.
        if self.enable_reverse_mamba and (self.self_attn is not None) and self._is_mamba_global():
            self.self_attn_reverse = Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)
            if self.fusion_mode == 'gated':
                self.gate_layer = nn.Linear(dim_h * 2, dim_h)
            elif self.fusion_mode == 'concat':
                self.concat_proj = nn.Linear(dim_h * 2, dim_h)
        else:
            self.self_attn_reverse = None
        # The structural edge-only variant owns its own forward/reverse Mambas.
        # It still needs the same directional fusion as the full edge branch.
        if global_model_type == 'Mamba_Edge' and scan_target == 'edge' and self.enable_reverse_mamba:
            if self.fusion_mode == 'gated':
                self.gate_layer = nn.Linear(dim_h * 2, dim_h)
            elif self.fusion_mode == 'concat':
                self.concat_proj = nn.Linear(dim_h * 2, dim_h)

        # Node/edge fusion when scan_target == 'both'.
        self.edge_node_fusion_mode = 'fixed'
        self.edge_node_weight = 0.5
        self.edge_node_gate_layer = None
        self.edge_node_concat_proj = None
        if self.scan_target == 'both':
            # Prefer dedicated keys if they exist; otherwise reuse the existing
            # reverse-fusion hyperparameters so users can control behavior
            # without introducing new config fields.
            try:
                from torch_geometric.graphgym.config import cfg as _cfg  # type: ignore

                _gt = getattr(_cfg, 'gt', _cfg)
                self.edge_node_fusion_mode = getattr(_gt, 'edge_node_fusion_mode', getattr(_gt, 'fusion_mode', 'fixed'))
                self.edge_node_weight = float(getattr(_gt, 'edge_node_weight', getattr(_gt, 'fixed_weight', 0.5)))
            except Exception:
                self.edge_node_fusion_mode = self.fusion_mode
                self.edge_node_weight = float(self.fixed_weight)

            if self.edge_node_fusion_mode == 'gated':
                self.edge_node_gate_layer = nn.Linear(dim_h * 2, dim_h)
            elif self.edge_node_fusion_mode == 'concat':
                self.edge_node_concat_proj = nn.Linear(dim_h * 2, dim_h)

        if edge_node_fusion_mode is not None:
            self.edge_node_fusion_mode = edge_node_fusion_mode
            if self.scan_target == 'both' and edge_node_fusion_mode == 'gated' and self.edge_node_gate_layer is None:
                self.edge_node_gate_layer = nn.Linear(dim_h * 2, dim_h)
            if self.scan_target == 'both' and edge_node_fusion_mode == 'concat' and self.edge_node_concat_proj is None:
                self.edge_node_concat_proj = nn.Linear(dim_h * 2, dim_h)
        self.edge_mamba = None
        self.edge_mamba_reverse = None
        if edge_node_weight is not None:
            self.edge_node_weight = float(edge_node_weight)
        self.structural_edge_scan = global_model_type == 'Mamba_Edge'
        self.structural_single_scan = self.structural_edge_scan and self.scan_target in ('node', 'edge')
        if self.edge_node_fusion_mode == 'single' and not self.structural_single_scan:
            raise ValueError('single fusion requires Mamba_Edge with scan_target=node or edge')
        if self.structural_edge_scan and not (
                self.node_edge_scan and (
                    (self.structural_single_scan and self.edge_node_fusion_mode == 'single')
                    or (self.scan_target == 'both' and self.edge_node_fusion_mode in
                        ('mfsg', 'sum', 'concat', 'necg_no_channel', 'necg_no_node')))):
            raise ValueError('Mamba_Edge requires a node_edge order and either both + NECG fusion, or node/edge + single fusion')
        self.stable_dual_scan = (
            self.scan_target == 'both'
            and global_model_type in ('Mamba_Hybrid_Degree_Noise', 'Mamba_Edge')
            and self.node_edge_scan
            and self.edge_node_fusion_mode in
            ('mfsg', 'concat', 'weighted', 'sum', 'necg_no_channel', 'necg_no_node'))
        if self.edge_node_fusion_mode in ('sum', 'necg_no_channel', 'necg_no_node') and not self.stable_dual_scan:
            raise ValueError('NECG ablations require the stable dual-scan architecture')
        if self.edge_node_fusion_mode in ('mfsg', 'weighted') and not self.stable_dual_scan:
            raise ValueError('mfsg/weighted require both Hybrid_Degree_Noise + node_edge_dfs, or Mamba_Edge + mfsg with a node_edge scan')
        if self.edge_node_fusion_mode == 'weighted' and not 0 <= self.edge_node_weight <= 1:
            raise ValueError('edge_node_weight must be in [0, 1]')
        self.stable_edge_scan = self.stable_dual_scan or (self.structural_single_scan and self.scan_target == 'edge')
        if self.stable_edge_scan:
            self.edge_mamba = Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)
            if self.enable_reverse_mamba:
                self.edge_mamba_reverse = Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)
            self.edge_mamba_input_norm = StableLayerNorm(dim_h)
            self.edge_mamba_update_norm = StableLayerNorm(dim_h)
            self.edge_mamba_output_proj = nn.Linear(dim_h, dim_h, bias=False)
            self.edge_state_norm = StableLayerNorm(dim_h)
        if self.stable_dual_scan or (self.structural_single_scan and self.scan_target == 'node'):
            self.mfsg_node_norm = StableLayerNorm(dim_h)
        if self.stable_edge_scan:
            self.mfsg_edge_norm = StableLayerNorm(dim_h)
        if self.edge_node_fusion_mode == 'mfsg':
            self.mfsg_channel_gate = nn.Sequential(
                nn.Linear(dim_h, max(1, dim_h // 4)), nn.GELU(),
                nn.Linear(max(1, dim_h // 4), dim_h), nn.Sigmoid())
            self.mfsg_spatial_gate = nn.Sequential(nn.Linear(dim_h, 1), nn.Sigmoid())
            # Start both interactions at exactly 0.5, then learn from data.
            nn.init.zeros_(self.mfsg_channel_gate[2].weight)
            nn.init.zeros_(self.mfsg_channel_gate[2].bias)
            nn.init.zeros_(self.mfsg_spatial_gate[0].weight)
            nn.init.zeros_(self.mfsg_spatial_gate[0].bias)
        # Keep the full MFSG initialization above unchanged, including RNG order.
        elif self.edge_node_fusion_mode == 'necg_no_channel':
            self.mfsg_spatial_gate = nn.Sequential(nn.Linear(dim_h, 1), nn.Sigmoid())
            nn.init.zeros_(self.mfsg_spatial_gate[0].weight)
            nn.init.zeros_(self.mfsg_spatial_gate[0].bias)
        elif self.edge_node_fusion_mode == 'necg_no_node':
            self.mfsg_channel_gate = nn.Sequential(
                nn.Linear(dim_h, max(1, dim_h // 4)), nn.GELU(),
                nn.Linear(max(1, dim_h // 4), dim_h), nn.Sigmoid())
            nn.init.zeros_(self.mfsg_channel_gate[2].weight)
            nn.init.zeros_(self.mfsg_channel_gate[2].bias)

    @staticmethod
    def _is_valid_permutation_1d(order: torch.Tensor, n: int) -> bool:
        """Fast validation: True iff `order` is a permutation of [0..n-1].

        Note: This is performance-critical (called inside forward); avoid Python
        loops / `.tolist()`.
        """
        if not torch.is_tensor(order):
            return False
        order = order.view(-1)
        if n == 0:
            return order.numel() == 0
        if order.numel() != n:
            return False
        if order.dtype != torch.long:
            # Still allow int tensors.
            if not order.dtype.is_floating_point:
                order = order.to(torch.long)
            else:
                return False
        # Range check.
        if int(order.min()) < 0 or int(order.max()) >= n:
            return False
        # Uniqueness check.
        # Sort+compare is usually faster than `unique` for this use case.
        sorted_vals = torch.sort(order).values
        target = torch.arange(n, device=sorted_vals.device, dtype=sorted_vals.dtype)
        return bool((sorted_vals == target).all())

    def _is_mamba_global(self) -> bool:
        return 'mamba' in self.global_model_type.lower()

    def forward(self, batch):
        # Defensive sanitize.
        h = torch.nan_to_num(batch.x, nan=0.0, posinf=1e4, neginf=-1e4)
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            batch.edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)

        h_in1 = h
        h_out_list = []

        # Local MPNN.
        if self.local_model is not None:
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = batch.pe_EquivStableLapPE if self.equivstable_pe else None
                local_out = self.local_model(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                )
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.equivstable_pe:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr, batch.pe_EquivStableLapPE)
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local

            h_local = torch.nan_to_num(h_local, nan=0.0, posinf=1e4, neginf=-1e4)
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                batch.edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Preserve local state without detaching gradients. The scan replaces
        # batch.edge_attr; current-layer aggregation/fusion still use its output.
        edge_local = batch.edge_attr if self.edge_state_transfer == 'local_only' else None

        # Global sequence model.
        if self.self_attn is not None or self.edge_mamba is not None:
            if self.scan_target == 'edge':
                if not self._is_mamba_global():
                    raise ValueError("scan_target='edge' currently supports only Mamba-based global_model_type.")
                h_attn = self._edge_mamba_scan(batch, h)
                if self.structural_single_scan:
                    h_attn = self.mfsg_edge_norm(h_attn)
            elif self.scan_target == 'node':
                with self._performance_region('gps/node_scan'):
                    h_attn = self._node_global_scan(batch, h)
                if self.structural_single_scan:
                    h_attn = self.mfsg_node_norm(h_attn)
            else:  # 'both'
                if not self._is_mamba_global():
                    raise ValueError("scan_target='both' requires a Mamba-based global_model_type.")
                with self._performance_region('gps/node_scan'):
                    h_node = self._node_global_scan(batch, h)
                with self._performance_region('gps/edge_branch'):
                    h_edge = self._edge_mamba_scan(batch, h)
                if self.edge_node_fusion_mode == 'mfsg':
                    with self._performance_region('gps/mfsg_fuse'):
                        h_attn = self._mfsg_fuse(h_node, h_edge, batch.batch)
                elif self.edge_node_fusion_mode in ('necg_no_channel', 'necg_no_node'):
                    h_attn = self._necg_ablation_fuse(h_node, h_edge, batch.batch)
                else:
                    h_attn = self._fuse_edge_node_outputs(h_node, h_edge)

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn
            h_attn = torch.nan_to_num(h_attn, nan=0.0, posinf=1e4, neginf=-1e4)
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local/global.
        if len(h_out_list) == 0:
            h = h_in1
        else:
            h = sum(h_out_list)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

        # Feed Forward block.
        h = h + self._ff_block(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        if self.edge_state_transfer == 'local_only':
            batch.edge_attr = edge_local
        return batch

    def _performance_region(self, name):
        return torch.profiler.record_function(name) if getattr(self, '_perf_enabled', False) else nullcontext()

    def _record_layout(self, kind, dense, valid):
        sample = getattr(self, '_perf_layout', None)
        if sample is None:
            return
        key = (sample, kind)
        if getattr(self, '_perf_last_' + kind, None) == key:
            return
        setattr(self, '_perf_last_' + kind, key)
        tokens = dense.size(0) * dense.size(1)
        logging.info('Layout split=%s batch=%d kind=%s valid=%d sequences=%d max_len=%d dense=%d padding=%.2f%%',
                     sample[0], sample[1], kind, valid, dense.size(0), dense.size(1), tokens,
                     100 * (1 - valid / max(tokens, 1)))
        old = self._perf_layout_stats.get(kind, (0, 0, 0))
        self._perf_layout_stats[kind] = (old[0] + valid, old[1] + tokens, old[2] + 1)

    def _debug_tensor(self, stage, tensor, batch_index=None, force=False):
        """Log high precision statistics before sanitizers can hide the source."""
        if not self.numerical_debug or not torch.is_tensor(tensor):
            return False
        value = tensor.detach()
        finite = torch.isfinite(value)
        finite_values = value[finite]
        abs_max = float(finite_values.abs().max()) if finite_values.numel() else None
        nonfinite = not bool(finite.all())
        dangerous = abs_max is not None and abs_max >= self.numerical_debug_threshold
        if not (force or nonfinite or dangerous):
            return False

        summary = {
            'shape': tuple(value.shape), 'dtype': str(value.dtype),
            'nan': int(torch.isnan(value).sum()),
            'posinf': int(torch.isposinf(value).sum()),
            'neginf': int(torch.isneginf(value).sum()),
            'finite_abs_max': abs_max,
        }
        logging.error("Numerical diagnostic stage=%s summary=%s", stage, summary)
        if value.dim() == 2 and value.size(0):
            value64 = value.to(torch.float64)
            row_finite = torch.isfinite(value).all(dim=-1)
            safe64 = torch.where(torch.isfinite(value64), value64, torch.zeros_like(value64))
            row_abs_max = safe64.abs().amax(dim=-1)
            row_mean = safe64.mean(dim=-1)
            row_var = safe64.var(dim=-1, unbiased=False)
            flagged = (~row_finite) | (row_abs_max >= self.numerical_debug_threshold)
            indices = flagged.nonzero(as_tuple=False).view(-1)
            if indices.numel() == 0 and force:
                indices = torch.topk(row_abs_max, min(8, value.size(0))).indices
            for row in indices[:8]:
                row_id = int(row)
                graph_id = int(batch_index[row]) if batch_index is not None else None
                logging.error(
                    "  stage=%s row=%d graph=%s finite=%s abs_max64=%g mean64=%g var64=%g",
                    stage, row_id, graph_id, bool(row_finite[row]),
                    float(row_abs_max[row]), float(row_mean[row]), float(row_var[row]))
        return nonfinite

    def _mfsg_fuse(self, h_node, h_edge, node_batch):
        """MFSG-inspired feature interaction (not topology-matrix modulation).

        Global node context controls edge-view channels per graph; the modulated
        edge view then controls node locations. Gates are independent sigmoid
        gates, not complementary mixture weights. Outer forward adds residual.
        """
        if h_node.shape != h_edge.shape:
            raise ValueError("MFSG requires aligned node and edge-view features")
        if h_node.size(0) == 0:
            return h_node + h_edge
        self._debug_tensor('mfsg_node_norm.input', h_node, node_batch)
        self._debug_tensor('mfsg_edge_norm.input', h_edge, node_batch)
        for name, parameter in (
                ('mfsg_edge_norm.weight', self.mfsg_edge_norm.weight),
                ('mfsg_edge_norm.bias', self.mfsg_edge_norm.bias)):
            if self._debug_tensor(name, parameter):
                raise RuntimeError(f"Non-finite parameter detected at {name}")
        node = self.mfsg_node_norm(h_node)
        edge = self.mfsg_edge_norm(h_edge)
        context = scatter_mean_fallback(node, node_batch, int(node_batch.max()) + 1)
        channel_gate = self.mfsg_channel_gate(context)[node_batch]
        edge_mod = edge * channel_gate
        spatial_gate = self.mfsg_spatial_gate(edge_mod)
        node_mod = node * spatial_gate
        return node_mod + edge_mod

    def _necg_ablation_fuse(self, h_node, h_edge, node_batch):
        """Remove one gate by identity bypass; keep both view normalizations."""
        if h_node.shape != h_edge.shape:
            raise ValueError('NECG requires aligned node and edge-view features')
        if h_node.size(0) == 0:
            return h_node + h_edge
        node = self.mfsg_node_norm(h_node)
        edge = self.mfsg_edge_norm(h_edge)
        if self.edge_node_fusion_mode == 'necg_no_channel':
            return node * self.mfsg_spatial_gate(edge) + edge
        if self.edge_node_fusion_mode == 'necg_no_node':
            context = scatter_mean_fallback(node, node_batch, int(node_batch.max()) + 1)
            return node + edge * self.mfsg_channel_gate(context)[node_batch]
        raise ValueError('Expected a single-gate NECG ablation')

    def _fuse_edge_node_outputs(self, h_node: torch.Tensor, h_edge: torch.Tensor) -> torch.Tensor:
        if self.stable_dual_scan:
            if h_node.shape != h_edge.shape:
                raise ValueError('Fusion requires aligned node and edge-view features')
            node = self.mfsg_node_norm(h_node)
            edge = self.mfsg_edge_norm(h_edge)
            if self.edge_node_fusion_mode == 'sum':
                return node + edge
            if self.edge_node_fusion_mode == 'concat':
                return self.edge_node_concat_proj(torch.cat((node, edge), dim=-1))
            if self.edge_node_fusion_mode == 'weighted':
                return self.edge_node_weight * node + (1 - self.edge_node_weight) * edge
            raise ValueError('MFSG interaction must use _mfsg_fuse')
        # Be defensive: if NaN/Inf appears, gated fusion can propagate it into
        # subsequent layers and lead to hard-to-debug CUDA asserts later.
        h_node = torch.nan_to_num(h_node, nan=0.0, posinf=1e4, neginf=-1e4)
        h_edge = torch.nan_to_num(h_edge, nan=0.0, posinf=1e4, neginf=-1e4)

        if h_node.shape != h_edge.shape:
            raise ValueError(f"Node/Edge scan output shape mismatch: {h_node.shape} vs {h_edge.shape}")

        mode = getattr(self, 'edge_node_fusion_mode', 'fixed')
        w = float(getattr(self, 'edge_node_weight', 0.5))
        w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

        # For regression tasks (like peptides-structural), apply stronger numerical stabilization
        # Check if this is likely a regression task by examining tensor magnitudes
        h_node_max = h_node.abs().max()
        h_edge_max = h_edge.abs().max()
        is_regression_like = (h_node_max > 10.0) or (h_edge_max > 10.0)  # Heuristic threshold

        if is_regression_like:
            # Apply gradient clipping before fusion for regression tasks
            clip_value = 5.0  # Conservative clipping
            h_node = torch.clamp(h_node, -clip_value, clip_value)
            h_edge = torch.clamp(h_edge, -clip_value, clip_value)

        # Normalize both outputs to improve numerical stability when fusing
        # This prevents one branch from dominating due to magnitude differences
        h_node_norm = torch.norm(h_node, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        h_edge_norm = torch.norm(h_edge, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        h_node_normalized = h_node / h_node_norm
        h_edge_normalized = h_edge / h_edge_norm

        # Use the average norm for reconstruction
        avg_norm = (h_node_norm + h_edge_norm) / 2.0

        if mode == 'fixed':
            fused = w * h_node_normalized + (1.0 - w) * h_edge_normalized
            result = fused * avg_norm
        elif mode == 'gated':
            if self.edge_node_gate_layer is None:
                raise RuntimeError("edge_node_gate_layer is not initialized (mode='gated').")
            combined = torch.cat([h_node_normalized, h_edge_normalized], dim=-1)
            gate = torch.sigmoid(self.edge_node_gate_layer(combined))
            fused = gate * h_node_normalized + (1.0 - gate) * h_edge_normalized
            result = fused * avg_norm
        elif mode == 'concat':
            if self.edge_node_concat_proj is None:
                raise RuntimeError("edge_node_concat_proj is not initialized (mode='concat').")
            combined = torch.cat([h_node_normalized, h_edge_normalized], dim=-1)
            fused = self.edge_node_concat_proj(combined)
            # Restore magnitude after projection
            fused_norm = torch.norm(fused, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            result = fused / fused_norm * avg_norm
        else:
            raise ValueError(f"Unsupported edge_node_fusion_mode: {mode}")

        # Final safety check for regression tasks
        if is_regression_like:
            result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result

    def _node_global_scan(self, batch, h: torch.Tensor) -> torch.Tensor:
        """Compute the global (sequence) model output over node embeddings.

        This retains the large collection of Mamba_* ordering variants used by
        the provided configs.
        """
        if self.global_scale_mode != 'none':
            h = self._apply_global_scale(h)
        if self.global_model_type in ['Transformer', 'Performer', 'BigBird', 'Mamba']:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                return self._sa_block(h_dense, None, ~mask)[mask]
            if self.global_model_type == 'Performer':
                return self.self_attn(h_dense, mask=mask)[mask]
            if self.global_model_type == 'BigBird':
                return self.self_attn(h_dense, attention_mask=mask)
            return self.self_attn(h_dense)[mask]

        if self.global_model_type.lower() == 'mamba_tokenization':
            if not self.enable_reverse_mamba:
                raise ValueError("Mamba_Tokenization requires enable_reverse_mamba=True for bidirectional scan.")
            num_nodes = h.size(0)
            m = int(self.mamba_tokenization_m)
            if m <= 0:
                deg = degree(batch.edge_index[0], num_nodes).to(torch.float)
                h_ind_perm = lexsort([deg, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                result = self._apply_mamba_scan_with_permutation(
                    h, batch.batch, h_ind_perm, h_dense, mask, h_ind_perm_reverse, batch
                )
                return torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)

            edge_index = batch.edge_index
            hop_feats = [h]
            if edge_index.numel() == 0:
                zero = h.new_zeros((num_nodes, h.size(-1)))
                for _ in range(m):
                    hop_feats.append(zero)
            else:
                src, dst = edge_index[0], edge_index[1]
                cur = h
                for _ in range(m):
                    nxt = scatter_mean_fallback(cur[src], dst, dim_size=num_nodes)
                    nxt = torch.nan_to_num(nxt, nan=0.0, posinf=1e4, neginf=-1e4)
                    hop_feats.append(nxt)
                    cur = nxt

            hop_stack = torch.stack(hop_feats[::-1], dim=1)
            node_order = lexsort([torch.arange(num_nodes, device=h.device, dtype=torch.long), batch.batch])
            hop_stack = hop_stack[node_order]
            token_seq = hop_stack.reshape(num_nodes * (m + 1), h.size(-1))
            token_batch = batch.batch[node_order].repeat_interleave(m + 1)
            token_dense, mask = to_dense_batch(token_seq, token_batch)
            token_out = self._apply_mamba_bucket_scan(token_dense, mask, batch)
            token_out = torch.nan_to_num(token_out, nan=0.0, posinf=1e4, neginf=-1e4)
            token_out = token_out.reshape(num_nodes, m + 1, -1)
            node_out_in_order = token_out[:, -1, :]
            node_out = node_out_in_order.new_empty((num_nodes, node_out_in_order.size(-1)))
            node_out[node_order] = node_out_in_order
            return torch.nan_to_num(node_out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.global_model_type == 'Mamba_DFS':
            return self._apply_traversal_mamba_scan(
                batch=batch,
                h=h,
                order_attr_name='dfs_node_order',
                order_builder=self._dfs_node_order,
                model_name='Mamba_DFS',
            )

        if self.global_model_type == 'Mamba_BFS':
            return self._apply_traversal_mamba_scan(
                batch=batch,
                h=h,
                order_attr_name='bfs_node_order',
                order_builder=self._bfs_node_order,
                model_name='Mamba_BFS',
            )

        if self.global_model_type == 'Mamba_Random':
            num_nodes = h.size(0)
            h_ind_perm = permute_within_batch(batch.batch)
            h_ind_perm = self._sanitize_node_order(h_ind_perm, num_nodes, device=h.device)
            h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
            h_ind_perm_reverse = torch.argsort(h_ind_perm)
            result = self._apply_mamba_scan_with_permutation(
                h, batch.batch, h_ind_perm, h_dense, mask, h_ind_perm_reverse, batch
            )
            return torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.global_model_type in ('Mamba_Hybrid_Degree_Noise', 'Mamba_Edge'):
            topology = self._cached_topology_counts(batch, h.size(0)) if self.topology_cache else None
            deg = (topology['src_count'].to(torch.float) if topology is not None and topology['valid']
                   else degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float))
            if batch.split == 'train':
                deg_noise = torch.rand_like(deg)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                result = self._apply_mamba_scan_with_permutation(
                    h, batch.batch, h_ind_perm, h_dense, mask, h_ind_perm_reverse, batch
                )
                return torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
            mamba_arr = []
            for _ in range(5):
                deg_noise = torch.rand_like(deg)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                result = self._apply_mamba_scan_with_permutation(
                    h, batch.batch, h_ind_perm, h_dense, mask, h_ind_perm_reverse, batch
                )
                mamba_arr.append(torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4))
            return torch.stack(mamba_arr).mean(dim=0)

        if self.global_model_type == 'Mamba_Hybrid_Degree_Noise_Bucket':
            deg_ = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
            if batch.split == 'train':
                deg_noise = torch.rand_like(deg_)
                deg = deg_ + deg_noise
                indices_arr, emb_arr = [], []
                bucket_assign = torch.randint(0, self.NUM_BUCKETS, (deg.numel(),), device=deg.device)
                for i in range(self.NUM_BUCKETS):
                    ind_i = (bucket_assign == i).nonzero().view(-1)
                    h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                    h_ind_perm_i = ind_i[h_ind_perm_sort]
                    h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                    h_dense = self._apply_mamba_bucket_scan(h_dense, mask, batch)
                    h_dense = torch.nan_to_num(h_dense, nan=0.0, posinf=1e4, neginf=-1e4)
                    indices_arr.append(h_ind_perm_i)
                    emb_arr.append(h_dense)
                h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                return torch.cat(emb_arr)[h_ind_perm_reverse]
            mamba_arr = []
            for _ in range(5):
                deg_noise = torch.rand_like(deg_)
                deg = deg_ + deg_noise
                indices_arr, emb_arr = [], []
                bucket_assign = torch.randint(0, self.NUM_BUCKETS, (deg.numel(),), device=deg.device)
                for i in range(self.NUM_BUCKETS):
                    ind_i = (bucket_assign == i).nonzero().view(-1)
                    h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                    h_ind_perm_i = ind_i[h_ind_perm_sort]
                    h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                    h_dense = self._apply_mamba_bucket_scan(h_dense, mask, batch)
                    h_dense = torch.nan_to_num(h_dense, nan=0.0, posinf=1e4, neginf=-1e4)
                    indices_arr.append(h_ind_perm_i)
                    emb_arr.append(h_dense)
                h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                mamba_arr.append(torch.cat(emb_arr)[h_ind_perm_reverse])
            return torch.stack(mamba_arr).mean(dim=0)

        if self.global_model_type == 'Mamba_RWSE':
            deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
            # RWSE_sum shape: (num_nodes,)
            RWSE_sum = torch.sum(batch.pestat_RWSE, dim=1)

            num_nodes = h.size(0)

            def _process_perm(h_ind_perm):
                # run self_attn on permuted dense and map back to node order
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                flat = self.self_attn(h_dense)[mask]
                flat = torch.nan_to_num(flat, nan=0.0, posinf=1e4, neginf=-1e4)
                # place outputs back into node order
                h_out = flat.new_empty((num_nodes, flat.size(-1)))
                h_out[h_ind_perm] = flat
                return h_out, h_ind_perm

            def _process_perm_with_reverse(h_ind_perm):
                # forward
                h_fwd, _ = _process_perm(h_ind_perm)

                if not self.enable_reverse_mamba:
                    return h_fwd

                # build reverse permutation within each graph (preserves graph order)
                h_ind_perm_rev = self._reverse_perm_within_batch(h_ind_perm, batch.batch)

                h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch.batch[h_ind_perm_rev])
                reverse_mamba = getattr(self, 'self_attn_reverse', None)
                if reverse_mamba is None:
                    raise ValueError("enable_reverse_mamba=True requires self_attn_reverse")
                flat_rev = reverse_mamba(h_dense_rev)[mask_rev]
                flat_rev = torch.nan_to_num(flat_rev, nan=0.0, posinf=1e4, neginf=-1e4)

                h_rev = flat_rev.new_empty((num_nodes, flat_rev.size(-1)))
                h_rev[h_ind_perm_rev] = flat_rev

                # fuse forward / reverse per-node outputs
                return self._fuse_mamba_outputs(h_fwd, h_rev)

            if batch.split == 'train':
                # single randomized ordering within +/- 1 STD (original behavior)
                RWSE_noise = torch.std(RWSE_sum) * torch.randn(RWSE_sum.shape, device=RWSE_sum.device)
                h_ind_perm = lexsort([-RWSE_sum + RWSE_noise, batch.batch])
                # Optionally sanitize permutation (consistent with other branches)
                h_ind_perm = self._sanitize_node_order(h_ind_perm, num_nodes, device=h.device)
                result = _process_perm_with_reverse(h_ind_perm)
                result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                return result

            else:
                # eval / ensemble: average several randomized orderings
                mamba_arr = []
                for i in range(5):
                    RWSE_noise = torch.std(RWSE_sum) * torch.randn(RWSE_sum.shape, device=RWSE_sum.device)
                    h_ind_perm = lexsort([-RWSE_sum + RWSE_noise, batch.batch])
                    h_ind_perm = self._sanitize_node_order(h_ind_perm, num_nodes, device=h.device)
                    h_out = _process_perm_with_reverse(h_ind_perm)
                    mamba_arr.append(torch.nan_to_num(h_out, nan=0.0, posinf=1e4, neginf=-1e4))
                # mean over attempts
                h_attn = sum(mamba_arr) / len(mamba_arr)
                return h_attn

        if self._is_mamba_global():
            h_dense, mask = to_dense_batch(h, batch.batch)
            return self.self_attn(h_dense)[mask]

        raise RuntimeError(f"Unexpected {self.global_model_type}")

    def _apply_traversal_mamba_scan(
            self,
            batch,
            h: torch.Tensor,
            order_attr_name: str,
            order_builder,
            model_name: str,
    ) -> torch.Tensor:
        node_order_attr = getattr(batch, order_attr_name, None)
        num_nodes = h.size(0)
        if node_order_attr is not None and self._is_valid_permutation_1d(node_order_attr, num_nodes):
            h_ind_perm = node_order_attr
        else:
            h_ind_perm = order_builder(batch.edge_index, batch.batch, num_nodes)
        h_ind_perm = self._sanitize_node_order(h_ind_perm, num_nodes, device=torch.device('cpu'))
        h_ind_perm = h_ind_perm.to(h.device, non_blocking=True)
        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
        flat_fwd = self.self_attn(h_dense)[mask]
        flat_fwd = torch.nan_to_num(flat_fwd, nan=0.0, posinf=1e4, neginf=-1e4)

        if flat_fwd.size(0) != num_nodes:
            if not hasattr(self, '_warned_bad_dense_mask'):
                setattr(self, '_warned_bad_dense_mask', True)
                warnings.warn(
                    f"{model_name}: mask sum ({flat_fwd.size(0)}) != num_nodes ({num_nodes}); falling back.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            h_dense0, mask0 = to_dense_batch(h, batch.batch)
            return self.self_attn(h_dense0)[mask0]

        h_fwd = flat_fwd.new_empty((num_nodes, flat_fwd.size(-1)))
        h_fwd[h_ind_perm] = flat_fwd
        h_fwd = torch.nan_to_num(h_fwd, nan=0.0, posinf=1e4, neginf=-1e4)
        if not self.enable_reverse_mamba:
            return h_fwd

        h_ind_perm_rev = torch.flip(h_ind_perm, dims=[0]).contiguous()
        h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch.batch[h_ind_perm_rev])
        flat_rev = self.self_attn_reverse(h_dense_rev)[mask_rev]
        flat_rev = torch.nan_to_num(flat_rev, nan=0.0, posinf=1e4, neginf=-1e4)
        if flat_rev.size(0) != num_nodes:
            return h_fwd
        h_rev = flat_rev.new_empty((num_nodes, flat_rev.size(-1)))
        h_rev[h_ind_perm_rev] = flat_rev
        h_rev = torch.nan_to_num(h_rev, nan=0.0, posinf=1e4, neginf=-1e4)
        return self._fuse_mamba_outputs(h_fwd, h_rev)

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return x

    def _ff_block(self, x):
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def _apply_global_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_scale_mode == 'none':
            return x
        if self.global_scale_mode == 'dim':
            scale = float(self.dim_h) ** 0.5
            return x / scale
        if self.global_scale_mode == 'fixed':
            return x * self.global_scale_value
        if self.global_scale_mode == 'learnable':
            if self.global_scale_param is None:
                raise RuntimeError("global_scale_param is not initialized for learnable scaling.")
            return x * self.global_scale_param
        raise ValueError(f"Unsupported global_scale_mode: {self.global_scale_mode}")

    def _fuse_mamba_outputs(self, h_fwd: torch.Tensor, h_rev: torch.Tensor) -> torch.Tensor:
        if h_fwd.shape != h_rev.shape:
            raise ValueError(f"Forward/Reverse Mamba output shape mismatch: {h_fwd.shape} vs {h_rev.shape}")

        if self.fusion_mode == 'fixed':
            return self.fixed_weight * h_fwd + (1 - self.fixed_weight) * h_rev
        if self.fusion_mode == 'gated':
            if self.gate_layer is None:
                raise RuntimeError("gate_layer is not initialized (mode='gated').")
            combined = torch.cat([h_fwd, h_rev], dim=-1)
            gate = torch.sigmoid(self.gate_layer(combined))
            return gate * h_fwd + (1 - gate) * h_rev
        if self.fusion_mode == 'concat':
            if self.concat_proj is None:
                raise RuntimeError("concat_proj is not initialized (mode='concat').")
            combined = torch.cat([h_fwd, h_rev], dim=-1)
            return self.concat_proj(combined)
        raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

    @staticmethod
    def _invalidate_topology_cache(batch):
        # DFS layouts depend on edge numbering as well as graph connectivity.
        for name in ('_gps_topology_cache', '_gps_dfs_paths_cache',
                     '_gps_edge_layout_cache', 'dfs_edge_order', 'dfs_edge_rank'):
            if hasattr(batch, name):
                delattr(batch, name)

    def _cached_topology_counts(self, batch, num_nodes):
        """Cache topology only; object/version checks detect relabels and edits.

        Inference tensors without mutation counters are deliberately not cached.
        Random permutations and learnable features must never enter this cache.
        """
        edges, nodes = batch.edge_index, batch.batch
        try:
            signature = (edges._version, nodes._version, tuple(edges.shape),
                         tuple(nodes.shape), edges.device, nodes.device,
                         edges.dtype, nodes.dtype, num_nodes)
        except RuntimeError:
            signature = None
        cached = getattr(batch, '_gps_topology_cache', None)
        if (signature is not None and cached is not None
                and cached[0] is edges and cached[1] is nodes and cached[2] == signature):
            return cached[3]
        if cached is not None:
            self._invalidate_topology_cache(batch)
        valid_mask = ((edges[0] >= 0) & (edges[0] < num_nodes)
                      & (edges[1] >= 0) & (edges[1] < num_nodes))
        valid = bool(valid_mask.all())
        stats = {'valid': valid, 'valid_mask': None if valid else valid_mask}
        if valid:
            stats['src_count'] = torch.bincount(edges[0], minlength=num_nodes)
            stats['dst_count'] = torch.bincount(edges[1], minlength=num_nodes)
        if signature is not None:
            batch._gps_topology_cache = (edges, nodes, signature, stats)
        return stats

    @staticmethod
    def _mean_with_counts(features, index, counts):
        # Keep the two directed means separate, including zero-degree nodes.
        result = features.new_zeros((counts.numel(), features.size(-1)))
        result.index_add_(0, index, features)
        return result / counts.to(dtype=features.dtype).clamp_min(1).unsqueeze(-1)

    def _project_edge_input(self, edge_attr, h, edge_index):
        """Project endpoints on nodes before gathering, using the original weight.

        No parameters are added or renamed: old model/optimizer states still fit.
        Scaling stays on inputs (not bias), preserving learnable scale gradients.
        """
        projection = self.edge_scan_input_proj
        if (self.edge_input_projection_optimized
                and self.node_edge_scan
                and isinstance(projection, nn.Linear)):
            if self.node_edge_scale_mode == 'none':
                node_input = h
            elif self.node_edge_scale_mode == 'dim':
                node_input = h / (float(self.dim_h) ** 0.5)
            elif self.node_edge_scale_mode == 'fixed':
                node_input = h * self.node_edge_scale_value
            elif self.node_edge_scale_mode == 'learnable':
                if self.node_edge_scale_param is None:
                    raise RuntimeError('node_edge_scale_param is not initialized')
                node_input = h * self.node_edge_scale_param
            else:
                raise ValueError(f'Unsupported node_edge_scale_mode: {self.node_edge_scale_mode}')
            if self.global_scale_mode != 'none':
                edge_attr = self._apply_global_scale(edge_attr)
                node_input = self._apply_global_scale(node_input)
            w_edge, w_src, w_dst = projection.weight.split(
                (edge_attr.size(-1), h.size(-1), h.size(-1)), dim=1)
            src, dst = edge_index
            result = F.linear(edge_attr, w_edge, projection.bias)
            result = result + F.linear(node_input, w_src)[src]
            return result + F.linear(node_input, w_dst)[dst]

        # Reference path also handles other scan orders / identity projections.
        if self.node_edge_scan:
            src, dst = edge_index
            if self.node_edge_scale_mode == 'none':
                node_src, node_dst = h[src], h[dst]
            elif self.node_edge_scale_mode == 'dim':
                scale = float(self.dim_h) ** 0.5
                node_src, node_dst = h[src] / scale, h[dst] / scale
            elif self.node_edge_scale_mode == 'fixed':
                node_src = h[src] * self.node_edge_scale_value
                node_dst = h[dst] * self.node_edge_scale_value
            elif self.node_edge_scale_mode == 'learnable':
                if self.node_edge_scale_param is None:
                    raise RuntimeError('node_edge_scale_param is not initialized')
                node_src = h[src] * self.node_edge_scale_param
                node_dst = h[dst] * self.node_edge_scale_param
            else:
                raise ValueError(f'Unsupported node_edge_scale_mode: {self.node_edge_scale_mode}')
            edge_attr = torch.cat([edge_attr, node_src, node_dst], dim=-1)
        if self.global_scale_mode != 'none':
            edge_attr = self._apply_global_scale(edge_attr)
        return projection(edge_attr)

    def _edge_mamba_scan(self, batch, h: torch.Tensor) -> torch.Tensor:
        if not self._is_mamba_global():
            raise ValueError("edge scan currently supports only Mamba-based global_model_type.")
        if not hasattr(batch, 'edge_attr') or batch.edge_attr is None:
            raise ValueError("edge scan requires batch.edge_attr.")
        if self.edge_scan_input_proj is None:
            raise RuntimeError("edge_scan_input_proj is not initialized.")

        edge_index = batch.edge_index
        num_nodes = h.size(0)

        # Ensure basic invariants early to avoid device-side asserts from
        # advanced indexing kernels.
        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(torch.long)
            batch.edge_index = edge_index

        # Keep edge_index and edge_attr consistent if something upstream
        # accidentally changed one but not the other.
        if batch.edge_attr is None:
            raise ValueError("edge scan requires batch.edge_attr.")
        if batch.edge_attr.size(0) != edge_index.size(1):
            min_len = min(int(batch.edge_attr.size(0)), int(edge_index.size(1)))
            if not hasattr(self, '_warned_edge_attr_mismatch'):
                setattr(self, '_warned_edge_attr_mismatch', True)
                warnings.warn(
                    f"edge_attr/edge_index length mismatch: edge_attr={batch.edge_attr.size(0)} vs "
                    f"num_edges={edge_index.size(1)}. Truncating to {min_len}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            batch.edge_attr = batch.edge_attr[:min_len]
            edge_index = edge_index[:, :min_len]
            batch.edge_index = edge_index
            self._invalidate_topology_cache(batch)

        topology = self._cached_topology_counts(batch, num_nodes) if self.topology_cache else None
        valid_edge_mask = topology['valid_mask'] if topology is not None else (
                (edge_index[0] >= 0)
                & (edge_index[0] < num_nodes)
                & (edge_index[1] >= 0)
                & (edge_index[1] < num_nodes)
        )
        valid = topology['valid'] if topology is not None else bool(valid_edge_mask.all())
        if not valid:
            edge_index = edge_index[:, valid_edge_mask]
            batch.edge_index = edge_index
            batch.edge_attr = batch.edge_attr[valid_edge_mask]
            self._invalidate_topology_cache(batch)
            if self.topology_cache:
                topology = self._cached_topology_counts(batch, num_nodes)

        edge_attr = batch.edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if not torch.is_floating_point(edge_attr):
            edge_attr = edge_attr.to(dtype=h.dtype)
        elif edge_attr.dtype != h.dtype:
            edge_attr = edge_attr.to(dtype=h.dtype)

        with self._performance_region('edge/input'):
            if (self.node_edge_scan
                    and not (self.edge_input_projection_optimized
                             and isinstance(self.edge_scan_input_proj, nn.Linear))):
                src, dst = edge_index[0], edge_index[1]
                if self.node_edge_scale_mode == 'none':
                    node_src = h[src]
                    node_dst = h[dst]
                elif self.node_edge_scale_mode == 'dim':
                    scale = float(self.dim_h) ** 0.5
                    node_src = h[src] / scale
                    node_dst = h[dst] / scale
                elif self.node_edge_scale_mode == 'fixed':
                    scale = self.node_edge_scale_value
                    node_src = h[src] * scale
                    node_dst = h[dst] * scale
                elif self.node_edge_scale_mode == 'learnable':
                    if self.node_edge_scale_param is None:
                        raise RuntimeError("node_edge_scale_param is not initialized for learnable scaling.")
                    node_src = h[src] * self.node_edge_scale_param
                    node_dst = h[dst] * self.node_edge_scale_param
                else:
                    raise ValueError(f"Unsupported node_edge_scale_mode: {self.node_edge_scale_mode}")
                edge_attr = torch.cat([edge_attr, node_src, node_dst], dim=-1)

            if (self.node_edge_scan
                    and self.edge_input_projection_optimized
                    and isinstance(self.edge_scan_input_proj, nn.Linear)):
                edge_feat = self._project_edge_input(edge_attr, h, edge_index)
            else:
                if self.global_scale_mode != 'none':
                    edge_attr = self._apply_global_scale(edge_attr)
                edge_feat = self.edge_scan_input_proj(edge_attr)
            edge_feat = torch.nan_to_num(edge_feat, nan=0.0, posinf=1e4, neginf=-1e4)
            if self.stable_edge_scan:
                if self.numerical_debug:
                    edge_input_batch = batch.batch[edge_index[0]] if edge_index.size(1) else None
                    self._debug_tensor('edge_mamba.input_projection', edge_feat, edge_input_batch)
                edge_feat = self.edge_mamba_input_norm(edge_feat)

        num_edges = int(edge_index.size(1))

        if self.edge_scan_order == 'default':
            edge_order = torch.arange(num_edges, device=edge_index.device)
        elif self.edge_scan_order == 'random':
            edge_order = torch.randperm(num_edges, device=edge_index.device)

        # Final safety: ensure edge_order is within bounds.
        if num_edges > 0 and self.edge_scan_order not in ['dfs', 'node_edge_dfs', 'node_edge_bfs', 'node_edge_random']:
            if edge_order.dtype != torch.long:
                edge_order = edge_order.to(torch.long)
            if edge_order.numel() != num_edges or int(edge_order.min()) < 0 or int(edge_order.max()) >= num_edges:
                edge_order = self._sanitize_edge_order(edge_order, num_edges, device=edge_index.device)

        if num_edges == 0:
            edge_out = edge_feat.new_empty((0, edge_feat.size(-1)))
        else:
            if self.edge_scan_order in ['dfs', 'node_edge_dfs', 'node_edge_bfs', 'node_edge_random']:
                with self._performance_region('edge/dfs'):
                    edge_order, edge_path_id = self._cached_dfs_edge_paths(
                        batch=batch,
                        edge_index=edge_index,
                        node_batch=batch.batch,
                        num_nodes=num_nodes,
                        max_path_len=512,
                    )

                edge_order = edge_order.to(device=h.device, dtype=torch.long)
                edge_path_id = edge_path_id.to(device=h.device, dtype=torch.long)

            else:
                edge_order = edge_order.to(device=h.device, dtype=torch.long)
                edge_batch_all = batch.batch[edge_index[0]]
                edge_path_id = edge_batch_all[edge_order].to(device=h.device, dtype=torch.long)

            with self._performance_region('edge/pack'):
                edge_feat_perm = edge_feat[edge_order]

                if self.edge_scan_order in ['dfs', 'node_edge_dfs', 'node_edge_bfs', 'node_edge_random']:
                    inverse_order, dense_positions, edge_mask = self._cached_edge_layout(
                        batch, edge_order, edge_path_id)
                    flat_dense = edge_feat.new_zeros((edge_mask.numel(), edge_feat.size(-1)))
                    flat_dense.index_copy_(0, dense_positions, edge_feat_perm)
                    edge_dense = flat_dense.view(*edge_mask.shape, edge_feat.size(-1))
                else:
                    edge_dense, edge_mask = to_dense_batch(edge_feat_perm, edge_path_id)
                    inverse_order = torch.argsort(edge_order)
            self._record_layout('edge', edge_dense, edge_feat_perm.size(0))

            with self._performance_region('edge/mamba'):
                edge_out_flat = self._apply_mamba_bucket_scan(edge_dense, edge_mask, batch, edge_scan=True)
            # _apply_mamba_bucket_scan already sanitizes its returned values.

            # 还原回原始 edge 顺序
            with self._performance_region('edge/restore'):
                edge_out = edge_out_flat[inverse_order]
        if self.stable_edge_scan:
            # Normalize the scan update before it enters the persistent edge
            # stream, then apply it as a controlled residual correction.
            with self._performance_region('edge/stabilize'):
                edge_update = self.edge_mamba_update_norm(edge_out)
                edge_update = self.edge_mamba_output_proj(edge_update)
                edge_out = self.edge_state_norm(
                    edge_feat + self.edge_mamba_residual_scale * edge_update)
            if self._debug_tensor('edge_state.after_residual_norm', edge_out):
                raise RuntimeError("Non-finite normalized edge state")
        batch.edge_attr = edge_out

        with self._performance_region('edge/aggregate'):
            src, dst = edge_index[0], edge_index[1]
            if self.topology_cache and topology is not None and topology['valid']:
                node_msg = self._mean_with_counts(edge_out, src, topology['src_count'])
                node_msg = node_msg + self._mean_with_counts(
                    edge_out, dst, topology['dst_count'])
            else:
                node_msg = scatter_mean_fallback(edge_out, src, dim_size=num_nodes)
                node_msg = node_msg + scatter_mean_fallback(edge_out, dst, dim_size=num_nodes)
            # Scale down to prevent gradient explosion when combining multiple sources
            node_msg = node_msg / 2.0
        if self._debug_tensor('edge_to_node_aggregation.raw', node_msg, batch.batch):
            raise RuntimeError("Non-finite raw edge-to-node aggregation output")
        node_msg = torch.nan_to_num(node_msg, nan=0.0, posinf=1e4, neginf=-1e4)
        return node_msg

    def _sanitize_edge_order(self, edge_order, num_edges: int, device: torch.device) -> torch.Tensor:
        if num_edges == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if not torch.is_tensor(edge_order):
            edge_order = torch.tensor(edge_order, dtype=torch.long)
        edge_order = edge_order.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        if edge_order.numel() == num_edges:
            in_range = (edge_order >= 0) & (edge_order < num_edges)
            if bool(in_range.all()):
                seen = torch.zeros(num_edges, dtype=torch.bool, device=device)
                seen[edge_order] = True
                if bool(seen.all()):
                    return edge_order

        if not hasattr(self, "_warned_invalid_edge_order"):
            setattr(self, "_warned_invalid_edge_order", True)
            warnings.warn(
                "Invalid edge_order detected; falling back to a sanitized permutation.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Slow-path: build a valid permutation.
        # Prefer tensor ops; avoid Python per-element loops.
        edge_order = edge_order.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_order = edge_order[(edge_order >= 0) & (edge_order < num_edges)]
        if edge_order.numel() == 0:
            return torch.arange(num_edges, device=device, dtype=torch.long)
        # Remove duplicates while preserving first occurrence.
        # Use CPU only when absolutely necessary.
        try:
            uniq, inv = torch.unique(edge_order, return_inverse=True)
            # `torch.unique` does not preserve order; emulate stable unique via
            # first occurrence positions.
            first_pos = torch.full((uniq.numel(),), edge_order.numel(), device=device, dtype=torch.long)
            pos = torch.arange(edge_order.numel(), device=device, dtype=torch.long)
            first_pos.scatter_reduce_(0, inv, pos, reduce='amin', include_self=True)
            keep = torch.argsort(first_pos)
            stable_uniq = uniq[keep]
        except Exception:
            stable_uniq = edge_order.detach().cpu().unique(sorted=False).to(device=device)

        seen = torch.zeros(num_edges, device=device, dtype=torch.bool)
        seen[stable_uniq] = True
        missing = (~seen).nonzero().view(-1)
        return torch.cat([stable_uniq, missing], dim=0)[:num_edges]

    def _sanitize_node_order(self, node_order, num_nodes: int, device: torch.device) -> torch.Tensor:
        if num_nodes == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if not torch.is_tensor(node_order):
            node_order = torch.tensor(node_order, dtype=torch.long)
        node_order = node_order.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        if node_order.numel() == num_nodes:
            in_range = (node_order >= 0) & (node_order < num_nodes)
            if bool(in_range.all()):
                seen = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                seen[node_order] = True
                if bool(seen.all()):
                    return node_order

        if not hasattr(self, "_warned_invalid_node_order"):
            setattr(self, "_warned_invalid_node_order", True)
            warnings.warn(
                "Invalid node_order detected; falling back to a sanitized permutation.",
                RuntimeWarning,
                stacklevel=2,
            )
        node_order = node_order.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        node_order = node_order[(node_order >= 0) & (node_order < num_nodes)]
        if node_order.numel() == 0:
            return torch.arange(num_nodes, device=device, dtype=torch.long)
        try:
            uniq, inv = torch.unique(node_order, return_inverse=True)
            first_pos = torch.full((uniq.numel(),), node_order.numel(), device=device, dtype=torch.long)
            pos = torch.arange(node_order.numel(), device=device, dtype=torch.long)
            first_pos.scatter_reduce_(0, inv, pos, reduce='amin', include_self=True)
            keep = torch.argsort(first_pos)
            stable_uniq = uniq[keep]
        except Exception:
            stable_uniq = node_order.detach().cpu().unique(sorted=False).to(device=device)

        seen = torch.zeros(num_nodes, device=device, dtype=torch.bool)
        seen[stable_uniq] = True
        missing = (~seen).nonzero().view(-1)
        return torch.cat([stable_uniq, missing], dim=0)[:num_nodes]

    def _dfs_node_order(self, edge_index: torch.Tensor, node_batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
        edge_index_cpu = edge_index.detach().cpu()
        node_batch_cpu = node_batch.detach().cpu()
        src = edge_index_cpu[0].tolist()
        dst = edge_index_cpu[1].tolist()
        adjacency = [[] for _ in range(num_nodes)]
        for u, v in zip(src, dst):
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adjacency[u].append(v)

        order = []
        unique_graphs = torch.unique(node_batch_cpu).tolist()
        for gid in unique_graphs:
            nodes = torch.where(node_batch_cpu == gid)[0].tolist()
            visited = set()
            node_set = set(nodes)
            for start in nodes:
                if start in visited:
                    continue
                stack = [start]
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    order.append(cur)
                    for nxt in reversed(adjacency[cur]):
                        if nxt in node_set and nxt not in visited:
                            stack.append(nxt)
            for n in nodes:
                if n not in visited:
                    order.append(n)
        return torch.tensor(order, device=edge_index.device, dtype=torch.long)

    def _bfs_node_order(self, edge_index: torch.Tensor, node_batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
        edge_index_cpu = edge_index.detach().cpu()
        node_batch_cpu = node_batch.detach().cpu()
        src = edge_index_cpu[0].tolist()
        dst = edge_index_cpu[1].tolist()
        adjacency = [[] for _ in range(num_nodes)]
        for u, v in zip(src, dst):
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adjacency[u].append(v)

        order = []
        unique_graphs = torch.unique(node_batch_cpu).tolist()
        for gid in unique_graphs:
            nodes = torch.where(node_batch_cpu == gid)[0].tolist()
            visited = set()
            node_set = set(nodes)
            for start in nodes:
                if start in visited:
                    continue
                queue = deque([start])
                visited.add(start)
                while queue:
                    cur = queue.popleft()
                    order.append(cur)
                    for nxt in adjacency[cur]:
                        if nxt in node_set and nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)
            for n in nodes:
                if n not in visited:
                    order.append(n)
        return torch.tensor(order, device=edge_index.device, dtype=torch.long)

    def _dfs_edge_order(self, edge_index: torch.Tensor, node_batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        edge_ids = list(range(edge_index.size(1)))
        adjacency = [[] for _ in range(num_nodes)]
        for eid, (u, v) in enumerate(zip(src, dst)):
            adjacency[u].append((v, eid))

        order = []
        unique_graphs = torch.unique(node_batch).tolist()
        for gid in unique_graphs:
            nodes = torch.where(node_batch == gid)[0].tolist()
            if not nodes:
                continue
            visited_nodes = set()
            used_edges = set()
            for start in nodes:
                if start in visited_nodes:
                    continue
                stack = [start]
                while stack:
                    cur = stack.pop()
                    if cur in visited_nodes:
                        continue
                    visited_nodes.add(cur)
                    for nxt, eid in reversed(adjacency[cur]):
                        if eid not in used_edges:
                            order.append(eid)
                            used_edges.add(eid)
                        if nxt not in visited_nodes and node_batch[nxt].item() == gid:
                            stack.append(nxt)

        if len(order) < len(edge_ids):
            used = set(order)
            order.extend([eid for eid in edge_ids if eid not in used])
        return torch.tensor(order, device=edge_index.device, dtype=torch.long)

    def _cached_edge_layout(self, batch, order, path_id):
        """Cache inverse permutation and dense positions, never feature values."""
        try:
            versions = (order._version, path_id._version)
        except RuntimeError:
            versions = None
        cached = getattr(batch, '_gps_edge_layout_cache', None)
        if (versions is not None and cached is not None
                and cached[0] is order and cached[1] is path_id
                and cached[2] == versions):
            return cached[3:]
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        _, mask = to_dense_batch(order.new_zeros((order.numel(), 1)), path_id)
        positions = torch.arange(mask.numel(), device=order.device)[mask.reshape(-1)]
        if versions is not None:
            batch._gps_edge_layout_cache = (order, path_id, versions, inverse, positions, mask)
        return inverse, positions, mask

    def _cached_dfs_edge_paths(self, batch, edge_index, node_batch,
                               num_nodes, max_path_len=512):
        """Reuse topology only within a batch; feature updates remain independent."""
        if self.structural_edge_scan:
            cached = getattr(batch, '_gps_structural_dfs', None)
            if cached is None or max_path_len != 512:
                raise ValueError('Mamba_Edge requires fresh structural scan metadata from GPSModel')
            tensors = (cached[0], cached[1], edge_index, node_batch)
            try:
                signature = tuple((t._version, tuple(t.shape), t.dtype, t.device)
                                  for t in tensors)
            except RuntimeError:
                signature = None  # inference tensors cannot track mutation
            checked = getattr(batch, '_gps_structural_validation', None)
            unchanged = (signature is not None and checked is not None
                         and checked[0] is cached and checked[2] == signature
                         and all(a is b for a, b in zip(checked[1], tensors)))
            if not unchanged:
                if (not torch.equal(cached[0], edge_index)
                        or not torch.equal(cached[1], node_batch)):
                    raise ValueError('Mamba_Edge requires fresh structural scan metadata from GPSModel')
                batch._gps_structural_validation = (cached, tensors, signature)
            return cached[2], cached[3]
        try:
            signature = (edge_index._version, node_batch._version,
                         tuple(edge_index.shape), tuple(node_batch.shape),
                         edge_index.device, node_batch.device, num_nodes, max_path_len)
        except RuntimeError:
            # Inference tensors have no mutation counter: do not cache them.
            signature = None
        cached = getattr(batch, '_gps_dfs_paths_cache', None)
        if (signature is not None and cached is not None
                and cached[0] is edge_index and cached[1] is node_batch
                and cached[2] == signature):
            if getattr(self, '_perf_cache_enabled', False):
                self._dfs_cache_hits += 1
            return cached[3], cached[4]
        if getattr(self, '_perf_cache_enabled', False):
            self._dfs_cache_misses += 1
        precomputed = assemble_dfs_scan_paths(
            batch, edge_index, node_batch, num_nodes, max_path_len)
        if precomputed is None:
            order, path_id = self._dfs_edge_paths(
                edge_index, node_batch, num_nodes, max_path_len)
            if getattr(self, '_perf_cache_enabled', False):
                self._dfs_runtime_builds = getattr(self, '_dfs_runtime_builds', 0) + 1
        else:
            order, path_id = precomputed
            if getattr(self, '_perf_cache_enabled', False):
                self._dfs_precomputed_uses = getattr(self, '_dfs_precomputed_uses', 0) + 1
        num_edges = edge_index.size(1)
        if (order.numel() != num_edges or path_id.numel() != num_edges
                or (num_edges and (int(order.min()) < 0 or int(order.max()) >= num_edges))):
            order = self._sanitize_edge_order(order, num_edges, edge_index.device)
            path_id = node_batch[edge_index[0]][order]
        if signature is not None:
            batch._gps_dfs_paths_cache = (edge_index, node_batch, signature, order, path_id)
        return order, path_id

    def _dfs_edge_paths(self, edge_index, node_batch, num_nodes, max_path_len=512):
        return dfs_edge_paths(edge_index, node_batch, num_nodes, max_path_len)[:2]

    def extra_repr(self):
        return (
            f"summary: dim_h={self.dim_h}, local_gnn_type={self.local_gnn_type}, "
            f"global_model_type={self.global_model_type}, heads={self.num_heads}"
        )

    def _apply_mamba_bucket_scan(self, h_dense: torch.Tensor, mask: torch.Tensor, batch, edge_scan=False) -> torch.Tensor:
        """Apply (optionally bidirectional) Mamba on a dense sequence and return flat masked output."""
        mamba = self.edge_mamba if edge_scan and self.edge_mamba is not None else self.self_attn
        h_fwd = mamba(h_dense)
        stage_prefix = 'edge_mamba' if edge_scan else 'node_mamba'
        if self.numerical_debug and self._debug_tensor(f'{stage_prefix}.forward.raw', h_fwd[mask]):
            raise RuntimeError(f"Non-finite raw output from {stage_prefix} forward scan")
        h_fwd = torch.nan_to_num(h_fwd, nan=0.0, posinf=1e4, neginf=-1e4)
        if not self.enable_reverse_mamba:
            return h_fwd[mask]
        reverse_mamba = getattr(self, 'self_attn_reverse', None)
        if edge_scan and self.edge_mamba_reverse is not None:
            reverse_mamba = self.edge_mamba_reverse
        if reverse_mamba is None:
            raise ValueError("enable_reverse_mamba=True requires self_attn_reverse")
        # Reverse valid tokens only: padding must never precede real tokens.
        positions = torch.arange(h_dense.size(1), device=h_dense.device)[None, :]
        lengths = mask.sum(dim=1, keepdim=True)
        reverse_index = torch.where(positions < lengths, lengths - 1 - positions, positions)
        reverse_index = reverse_index.unsqueeze(-1).expand_as(h_dense)
        h_rev = h_dense.gather(1, reverse_index)
        h_rev = reverse_mamba(h_rev)
        if self.numerical_debug and self._debug_tensor(f'{stage_prefix}.reverse.raw', h_rev[mask]):
            raise RuntimeError(f"Non-finite raw output from {stage_prefix} reverse scan")
        h_rev = h_rev.gather(1, reverse_index)
        h_rev = torch.nan_to_num(h_rev, nan=0.0, posinf=1e4, neginf=-1e4)
        h_attn = self._fuse_mamba_outputs(h_fwd, h_rev)
        if self.numerical_debug and self._debug_tensor(f'{stage_prefix}.bidirectional_fusion.raw', h_attn[mask]):
            raise RuntimeError(f"Non-finite raw output from {stage_prefix} bidirectional fusion")
        h_attn = torch.nan_to_num(h_attn, nan=0.0, posinf=1e4, neginf=-1e4)
        return h_attn[mask]

    def _apply_mamba_scan_with_permutation(
            self,
            h: torch.Tensor,
            batch_index: torch.Tensor,
            h_ind_perm: torch.Tensor,
            h_dense: torch.Tensor,
            mask: torch.Tensor,
            h_ind_perm_reverse: torch.Tensor,
            batch,
    ) -> torch.Tensor:
        """Run Mamba on a permuted dense sequence (and optionally reverse) and map back."""
        self._record_layout('node', h_dense, h.size(0))
        h_fwd = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
        h_fwd = torch.nan_to_num(h_fwd, nan=0.0, posinf=1e4, neginf=-1e4)
        if not self.enable_reverse_mamba:
            return h_fwd

        # IMPORTANT:
        # A naive `torch.flip(h_ind_perm)` makes `batch_index[h_ind_perm_rev]`
        # typically *decreasing* (graphs appear in reverse order). Some PyG
        # versions/utilities assume the `batch` vector is sorted/non-decreasing
        # and can produce an incorrect mask whose number of True entries is
        # smaller than `num_nodes`. That mismatch then triggers CUDA
        # IndexKernel out-of-bounds when we later index with an inverse
        # permutation of length `num_nodes`.
        #
        # Fix: reverse the node order *within each graph* while keeping graphs
        # in increasing batch id order.
        h_ind_perm_rev = self._reverse_perm_within_batch(h_ind_perm, batch_index)

        h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch_index[h_ind_perm_rev])
        h_ind_perm_rev_reverse = torch.argsort(h_ind_perm_rev)
        reverse_mamba = getattr(self, 'self_attn_reverse', None)
        if reverse_mamba is None:
            raise ValueError("enable_reverse_mamba=True requires self_attn_reverse")
        h_rev = reverse_mamba(h_dense_rev)[mask_rev][h_ind_perm_rev_reverse]
        h_rev = torch.nan_to_num(h_rev, nan=0.0, posinf=1e4, neginf=-1e4)
        result = self._fuse_mamba_outputs(h_fwd, h_rev)
        return torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)

    @staticmethod
    def _reverse_perm_within_batch(h_ind_perm: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Reverse a node permutation within each graph, preserving graph order.

        Args:
            h_ind_perm: A 1D index tensor (ideally a permutation of [0..N-1]).
            batch_index: The `batch.batch` vector mapping each node -> graph id.

        Returns:
            A new permutation where nodes are reversed within each graph.
        """
        if h_ind_perm.numel() == 0:
            return h_ind_perm
        if h_ind_perm.dtype != torch.long:
            h_ind_perm = h_ind_perm.to(torch.long)

        # Fully vectorized GPU implementation.
        device = h_ind_perm.device
        batch_perm = batch_index[h_ind_perm].to(torch.long)

        # Ensure (mostly) non-decreasing batch ids; if not, stably sort by batch.
        if batch_perm.numel() > 1 and not bool((batch_perm[1:] >= batch_perm[:-1]).all()):
            order = torch.argsort(batch_perm, stable=True)
            h_ind_perm = h_ind_perm[order]
            batch_perm = batch_perm[order]

        n = int(h_ind_perm.numel())
        if n == 0:
            return h_ind_perm

        num_graphs = int(batch_perm.max().item()) + 1
        counts = torch.bincount(batch_perm, minlength=num_graphs)
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts

        pos = torch.arange(n, device=device, dtype=torch.long)
        start_pos = starts[batch_perm]
        end_pos = offsets[batch_perm]
        pos_rev = start_pos + (end_pos - 1 - pos)
        return h_ind_perm[pos_rev]
