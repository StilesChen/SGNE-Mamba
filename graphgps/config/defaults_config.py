from torch_geometric.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = True

    # When saving the best checkpoint, also export split-level predictions
    # and targets to a single PyTorch artifact.
    cfg.train.save_predictions = True

    cfg.train.profile = False
    # Content-addressed, deterministic graph preprocessing shared across seeds.
    cfg.dataset.deterministic_preprocess_cache = True
    cfg.dataset.deterministic_preprocess_cache_dir = ''
    # Bounded CPU DFS preparation overlapping the previous batch's logging.
    # Off by default: benchmark on the training host before enabling.
    cfg.train.structural_dfs_prefetch = False
    # Performance measurements are independent of numerical diagnostics.
    cfg.train.performance_timing = False
    cfg.train.performance_layout_batches = 0
    cfg.train.performance_profile = False
    cfg.train.performance_profile_epoch = 2  # Zero-based; profile the third epoch.
    cfg.train.performance_profile_wait = 10
    cfg.train.performance_profile_warmup = 3
    cfg.train.performance_profile_active = 5

    # Numerical diagnostics. Gradient reporting is cheap and only emits output
    # after a non-finite gradient is found. Autograd anomaly detection is much
    # slower, so enable it temporarily when reproducing a failure.
    cfg.train.nan_diagnostics = True
    cfg.train.nan_diagnostics_max_params = 50
    cfg.train.detect_anomaly = False
    cfg.train.detect_nonfinite_forward = False
    cfg.train.numerical_debug = False
    cfg.train.numerical_debug_threshold = 1e12
