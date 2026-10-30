from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_gt')
def set_cfg_gt(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    # Positional encodings argument group
    cfg.gt = CN()

    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = 'SANLayer'

    # Number of Transformer layers in the model
    cfg.gt.layers = 3

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64

    # Size of the edge embedding
    cfg.gt.dim_edge = None

    # Full attention SAN transformer including all possible pairwise edges
    cfg.gt.full_graph = True

    # Type of extra edges used for transformer
    cfg.gt.secondary_edges = 'full_graph'

    # SAN real vs fake edge attention weighting coefficient
    cfg.gt.gamma = 1e-5

    # Histogram of in-degrees of nodes in the training set used by PNAConv.
    # Used when `gt.layer_type: PNAConv+...`. If empty it is precomputed during
    # the dataset loading process.
    cfg.gt.pna_degrees = []

    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    # Dropout in self-attention.
    cfg.gt.attn_dropout = 0.0

    cfg.gt.layer_norm = False

    cfg.gt.batch_norm = True

    cfg.gt.residual = True

    cfg.gt.activation = 'relu'

    # Enable reverse Mamba for bidirectional scanning
    cfg.gt.enable_reverse_mamba = True

    # Fusion mode for forward and reverse Mamba: 'fixed' or 'gated'
    cfg.gt.fusion_mode = 'fixed'

    # Fixed weight for forward Mamba in fusion (0.0 to 1.0)
    cfg.gt.fixed_weight = 0.7

    cfg.gt.scan_target = 'node'
    # Mamba_Edge single-scan capacity controls: node/edge with
    # edge_node_fusion_mode='single'. Preserve the selected branch's view norm;
    # remove the other scanner and cross-domain gates (no unused parameters).

    # default / dfs / node_edge_dfs / random; Mamba_Edge also: node_edge_bfs / node_edge_random
    cfg.gt.edge_scan_order = 'default'
    # Scaling for node features concatenated into edge_attr when using node_edge_dfs/bfs/random:
    # none / dim / fixed / learnable
    cfg.gt.node_edge_scale_mode = 'dim'
    # Scale value for fixed/learnable modes
    cfg.gt.node_edge_scale_value = 1.0
    # Scaling for inputs to the global sequence model (Mamba/Transformer/Performer/BigBird):
    # none / dim / fixed / learnable
    cfg.gt.global_scale_mode = 'none'
    # Scale value for fixed/learnable global scaling
    cfg.gt.global_scale_value = 1.0

    # When `scan_target: both`, fuse node-scan and edge-scan outputs.
    # fixed / gated / concat / mfsg / sum / necg_no_channel / necg_no_node.
    # Mamba_Edge NECG ablations keep the same independent stable scanners.
    # sum: LN(node)+LN(edge); concat: projected concatenation of normalized views.
    # necg_no_channel / necg_no_node bypass the named gate with identity (1).
    # MFSG (paper: NECG) uses independent node/edge Mambas;
    # requires both + Hybrid_Degree_Noise + node_edge_dfs, or
    # both + Mamba_Edge + node_edge_dfs/node_edge_bfs/node_edge_random.
    cfg.gt.edge_node_fusion_mode = 'fixed'
    # For both Hybrid_Degree_Noise + node_edge_dfs: mfsg / concat / weighted
    # use independent stabilized scanners. weighted: w * node + (1-w) * edge.

    cfg.gt.edge_node_weight = 0.5
    # Strength of the normalized Edge-Mamba update written to the next layer.
    cfg.gt.edge_mamba_residual_scale = 0.1
    # Inter-layer edge transfer: full / local_only / reset_initial.
    # All modes retain the current layer's edge scan and node fusion.
    cfg.gt.edge_state_transfer = 'full'
    # Precompute graph-local DFS paths once during dataset loading.
    cfg.gt.edge_scan_precompute = True
    # Equivalent execution optimizations; disable for reference comparisons.
    cfg.gt.edge_input_projection_optimized = True
    cfg.gt.topology_cache = True
    # Mamba_Edge: DFS/BFS use raw-attribute structural priorities; random shuffles
    # all edges within each graph. Sampled before each forward, shared by layers.
    cfg.gt.edge_dfs_signature_rounds = 2
    cfg.gt.edge_scan_segmentation = 'native'  # native / fixed_chunks
    cfg.gt.edge_scan_within_segment_shuffle = False
    cfg.gt.edge_scan_max_segment_len = 512
    cfg.gt.edge_dfs_use_attributes = True

    # Hop count for Mamba_Tokenization / Mamba_RWPE_Tokenization (m >= 0).
    cfg.gt.mamba_tokenization_m = 2

    # Mamba_RWPE_Tokenization hyperparams.
    cfg.gt.mamba_rwpe_num_walks = 4
    cfg.gt.mamba_rwpe_walk_length = 8
    cfg.gt.mamba_rwpe_subgraph_layers = 2
    cfg.gt.mamba_rwpe_fusion_weight = 0.5
    # If set, overrides the assumed input edge feature dim used for token induced subgraphs.
    cfg.gt.mamba_rwpe_edge_dim_in = None

    # BigBird model/GPS-BigBird layer.
    cfg.gt.bigbird = CN()

    cfg.gt.bigbird.attention_type = "block_sparse"

    cfg.gt.bigbird.chunk_size_feed_forward = 0

    cfg.gt.bigbird.is_decoder = False

    cfg.gt.bigbird.add_cross_attention = False

    cfg.gt.bigbird.hidden_act = "relu"

    cfg.gt.bigbird.max_position_embeddings = 128

    cfg.gt.bigbird.use_bias = False

    cfg.gt.bigbird.num_random_blocks = 3

    cfg.gt.bigbird.block_size = 3

    cfg.gt.bigbird.layer_norm_eps = 1e-6
