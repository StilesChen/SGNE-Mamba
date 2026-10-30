import logging
import os
import time
from graphgps.train.performance import measured_epoch, performance_region
from graphgps.train.structural_prefetch import with_structural_prefetch, StructuralPrefetchLoader
from graphgps.transform.structural_dfs import prepare_model_batch, activate_structural_dfs

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name

# from deepspeed.profiling.flops_profiler import FlopsProfiler
# from torch.autograd import profiler
from torch.profiler import profile, record_function, ProfilerActivity


def _finite_tensor_summary(tensor):
    """Return synchronization-friendly diagnostics for one tensor."""
    value = tensor.detach()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    return {
        'shape': tuple(value.shape),
        'dtype': str(value.dtype),
        'nan': int(torch.isnan(value).sum().item()),
        'posinf': int(torch.isposinf(value).sum().item()),
        'neginf': int(torch.isneginf(value).sum().item()),
        'finite_abs_max': (
            float(finite_values.abs().max().item())
            if finite_values.numel() else None
        ),
    }


def _has_nonfinite_gradients(model):
    """Reduce flags on each device before synchronizing with the host."""
    flags_by_device = {}
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        value = grad.detach()
        if value.is_sparse:
            value = value._values()
        flags_by_device.setdefault(value.device, []).append(
            torch.isfinite(value).all())
    return any(not bool(torch.stack(flags).all())
               for flags in flags_by_device.values())


def _report_nonfinite_gradients(model, epoch, iteration, loss, max_params=50):
    """Log every affected parameter instead of stopping at the first one."""
    if not _has_nonfinite_gradients(model):
        return False
    bad = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is not None and not torch.isfinite(grad).all():
            bad.append((name, _finite_tensor_summary(grad)))

    if not bad:
        return False

    logging.error(
        "Non-finite gradients at epoch=%s iteration=%s loss=%s; "
        "affected_parameters=%d",
        epoch, iteration, float(loss.detach().cpu()), len(bad),
    )
    limit = max(1, int(max_params))
    for name, summary in bad[:limit]:
        logging.error("  gradient %s: %s", name, summary)
    if len(bad) > limit:
        logging.error("  ... %d more affected parameters omitted", len(bad) - limit)

    # Parameters may already be corrupt from an earlier optimizer update.
    bad_values = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            bad_values.append((name, _finite_tensor_summary(param)))
    for name, summary in bad_values[:limit]:
        logging.error("  parameter value %s: %s", name, summary)
    return True


def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _install_nonfinite_forward_hooks(model):
    """Stop at the first leaf module that emits a NaN or Inf activation."""
    if getattr(model, '_nonfinite_forward_hook_handles', None):
        return

    handles = []
    for module_name, module in model.named_modules():
        if not module_name or any(module.children()):
            continue

        def check_output(_module, _inputs, output, name=module_name):
            for index, tensor in enumerate(_iter_tensors(output)):
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    summary = _finite_tensor_summary(tensor)
                    logging.error(
                        "First non-finite leaf-module output: module=%s "
                        "output_tensor=%d summary=%s",
                        name, index, summary,
                    )
                    raise RuntimeError(
                        f"Non-finite forward activation produced by module '{name}'"
                    )

        handles.append(module.register_forward_hook(check_output))
    model._nonfinite_forward_hook_handles = handles
    logging.warning(
        "Installed non-finite forward hooks on %d leaf modules. "
        "This debugging mode synchronizes device execution and is slow.",
        len(handles),
    )


def subsample_batch_index(batch, min_k=1, ratio=0.1):
    torch.manual_seed(0)
    unique_batches = torch.unique(batch.batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch.batch == batch_index).nonzero().squeeze()
        # See how many nodes in the graphs
        # And how many left after subsetting
        k = int(indices_in_batch.size(0) * ratio)
        # If subsetting gives more than 1, do subsetting
        if k > min_k:
            perm = torch.randperm(indices_in_batch.size(0))
            idx = perm[:k]
            idx = indices_in_batch[idx]
            idx, _ = torch.sort(idx)
        # Otherwise retain entire graph
        else:
            idx = indices_in_batch
        permuted_indices.append(idx)
    idx = torch.cat(permuted_indices)
    return idx


def arxiv_cross_entropy(pred, true, split_idx):
    true = true.squeeze(-1)
    if pred.ndim > 1 and true.ndim == 1:
        pred_score = F.log_softmax(pred[split_idx], dim=-1)
        loss = F.nll_loss(pred_score, true[split_idx])
    else:
        raise ValueError("In ogbn cross_entropy calculation dimensions did not match.")
    return loss, pred_score


def _detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _detach_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_detach_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(v) for v in value)
    return value


def _merge_prediction_batches(batches):
    if not batches:
        return []

    first = batches[0]
    if torch.is_tensor(first):
        return torch.cat(batches, dim=0)
    if isinstance(first, np.ndarray):
        return np.concatenate(batches, axis=0)
    if isinstance(first, dict):
        return {
            key: _merge_prediction_batches([batch[key] for batch in batches])
            for key in first
        }
    if isinstance(first, list):
        merged = []
        for batch in batches:
            merged.extend(batch)
        return merged
    if isinstance(first, tuple):
        merged = []
        for batch in batches:
            merged.extend(list(batch))
        return merged
    return batches


def _is_pcqm4m_contact_dataset():
    return cfg.dataset.name.startswith('PCQM4Mv2Contact')


def _resolve_inductive_edge_head(model):
    if hasattr(model, 'post_mp'):
        return model.post_mp
    if hasattr(model, 'model') and hasattr(model.model, 'post_mp'):
        return model.model.post_mp
    return None


def _collect_contact_predictions(batch, model):
    if cfg.model.edge_decoding != 'dot':
        raise ValueError(
            f"Unsupported edge decoding {cfg.model.edge_decoding} for MRR collection."
        )

    head = _resolve_inductive_edge_head(model)
    prepare_model_batch(model, batch)
    batch = batch.to(torch.device(cfg.device))
    batch_for_pred = batch.clone()
    activate_structural_dfs(batch_for_pred)
    # Ensure batch_for_pred has .split (some layers expect it)
    batch_for_pred.split = getattr(batch, 'split', 'val')

    if head is not None and cfg.model.edge_decoding != 'concat':
        # Ensure we operate on node embeddings of dim cfg.gnn.dim_inner.
        # Try to run model's front modules (encoder, pre_mp, layers) if present,
        # so batch_for_pred.x has expected dim (cfg.gnn.dim_inner).
        parent = model
        # If head nested inside model.model, pick that parent
        if not hasattr(parent, 'post_mp') and hasattr(model, 'model') and hasattr(model.model, 'post_mp'):
            parent = model.model

        # Run encoder, pre_mp, layers if available to produce internal embeddings
        for mod_name in ('encoder', 'pre_mp', 'layers'):
            if hasattr(parent, mod_name):
                module = getattr(parent, mod_name)
                try:
                    batch_for_pred = module(batch_for_pred)
                except Exception as e:
                    # Log shape to help debugging and re-raise
                    x_shape = None
                    if hasattr(batch_for_pred, 'x') and batch_for_pred.x is not None:
                        x_shape = tuple(batch_for_pred.x.shape)
                    raise RuntimeError(f"Failed to run {mod_name} before head: batch.x shape={x_shape}, error={e}")

        # Now dtype align and apply the post-mp head module
        head_params = list(head.layer_post_mp.parameters())
        if head_params:
            target_dtype = head_params[0].dtype
            if hasattr(batch_for_pred, 'x') and batch_for_pred.x is not None:
                batch_for_pred.x = batch_for_pred.x.to(target_dtype)

        batch_for_pred = head.layer_post_mp(batch_for_pred)

    pred_pos_list = []
    pred_neg_list = []

    for data in batch_for_pred.to_data_list():
        pred = data.x @ data.x.transpose(0, 1)
        pos_edge_index = data.edge_index_labeled[:, data.edge_label == 1]
        num_pos_edges = pos_edge_index.shape[1]

        pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]
        if num_pos_edges > 0:
            neg_mask = torch.ones([num_pos_edges, data.num_nodes], dtype=torch.bool)
            neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
            pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
        else:
            pred_neg = pred.new_empty((0, max(data.num_nodes - 1, 0)))

        pred_pos_list.append(pred_pos.detach().cpu())
        pred_neg_list.append(pred_neg.detach().cpu())

    return pred_pos_list, pred_neg_list


def _forward_eval_batch(loader, model, batch, split):
    batch.split = split
    prepare_model_batch(model, batch)
    batch.to(torch.device(cfg.device), non_blocking=True)
    if cfg.gnn.head == 'inductive_edge':
        pred, true, extra_stats = model(batch)
    else:
        pred, true = model(batch)
        extra_stats = {}

    if cfg.dataset.name == 'ogbg-code2':
        loss, pred_score = subtoken_cross_entropy(pred, true)
        _true = true
        _pred = pred_score
    elif cfg.dataset.name == 'ogbn-arxiv':
        index_split = loader.dataset.split_idx[split].to(torch.device(cfg.device))
        loss, pred_score = arxiv_cross_entropy(pred, true, index_split)
        _true = true[index_split].detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
    else:
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)

    return loss, _true, _pred, extra_stats


# @profiler.profile
# def profile_mem_forward(model, batch):
#     pred, true = model(batch)
#     return pred, true

@measured_epoch
@with_structural_prefetch
def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation, cur_epoch=None):
    # flop related
    if_mem = False
    if_flop = False
    if_select = False
    if if_flop:
        from deepspeed.profiling.flops_profiler import FlopsProfiler
        prof = FlopsProfiler(model, None)
        # profile_step = 0
        total_flop_s = 0.
        sample_count = 0
        if if_select:
            total_node = 0

    detect_anomaly = bool(getattr(cfg.train, 'detect_anomaly', False))
    torch.autograd.set_detect_anomaly(detect_anomaly, check_nan=True)
    if detect_anomaly and (cur_epoch is None or cur_epoch == 0):
        logging.warning(
            "Autograd anomaly detection is enabled. Training will be slower; "
            "the exception traceback identifies the backward operation and its forward call."
        )
    if getattr(cfg.train, 'detect_nonfinite_forward', False):
        _install_nonfinite_forward_hooks(model)

    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        if if_select:
            ratio = 1.0
            idx = subsample_batch_index(batch, min_k=1, ratio=ratio)
            batch = batch.subgraph(idx)
        # flop related
        if if_flop:  # and iter == profile_step:
            prof.start_profile()
        batch.split = 'train'
        if getattr(batch, '_gps_structural_prefetched', False) and not if_select:
            delattr(batch, '_gps_structural_prefetched')
        else:
            prepare_model_batch(model, batch)
        batch.to(torch.device(cfg.device), non_blocking=True)

        # 防止 model forward 中对 edge_attr 的原地修改污染下一个 epoch 的数据
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            # 额外的数值保护（可选）
            batch.edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)
            batch.edge_attr = torch.clamp(batch.edge_attr, -1e3, 1e3)

        pred, true = model(batch)
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        elif cfg.dataset.name == 'ogbn-arxiv':
            split_idx = loader.dataset.split_idx['train'].to(torch.device(cfg.device))
            loss, pred_score = arxiv_cross_entropy(pred, true, split_idx)
            _true = true[split_idx].detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)

        if if_flop:
            prof.stop_profile()
            flops = prof.get_total_flops()
            flops_s = flops / 1000000000.
            total_flop_s += flops_s
            sample_count += len(torch.unique(batch.batch))
            params = prof.get_total_params()
            prof.end_profile()
            if if_select:
                total_node += batch.x.size(0)

        # 防守：检查损失和预测是否为 NaN/Inf
        if not torch.isfinite(loss):
            logging.warning(f"Non-finite loss detected at iteration {iter}: {loss.item()}. Skipping backward pass.")
            optimizer.zero_grad()
            continue

        if not torch.isfinite(pred_score).all():
            num_bad = (~torch.isfinite(pred_score)).sum().item()
            logging.warning(f"Non-finite predictions detected at iteration {iter}: {num_bad}/{pred_score.numel()} elements.")

        with performance_region('train/backward'):
            loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            # Report all affected parameters. This complements anomaly mode:
            # anomaly mode identifies a failing op, while this identifies which
            # model branches received non-finite gradients.
            with performance_region('train/gradient_check'):
                if getattr(cfg.train, 'nan_diagnostics', True):
                    has_nan_grad = _report_nonfinite_gradients(
                        model=model,
                        epoch=cur_epoch,
                        iteration=iter,
                        loss=loss,
                        max_params=getattr(cfg.train, 'nan_diagnostics_max_params', 50),
                    )
                else:
                    has_nan_grad = _has_nonfinite_gradients(model)

            if has_nan_grad:
                if cur_epoch is not None:
                    save_ckpt(model, optimizer, scheduler, cur_epoch)
                logging.error(
                    "Non-finite gradients detected. Terminating training. "
                    "Set train.detect_anomaly: true to locate the originating autograd operation."
                )
                raise RuntimeError("Non-finite gradients detected during training")
            else:
                with performance_region('train/optimizer'):
                    if cfg.optim.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
        if isinstance(loader, StructuralPrefetchLoader):
            loader.prefetch_next()
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()
    if if_flop:
        print('################ Print flop')
        print(total_flop_s / sample_count, params)
        print('################ End print flop')
    if if_mem:
        print('################ Print mem')
        print(torch.cuda.max_memory_allocated() / (1024 ** 2))
        print(torch.cuda.max_memory_reserved() / (1024 ** 2))
        # print(prof_mem.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        print('################ End print mem')
    if if_select:
        print('################ Print avg nodes')
        print(total_node / sample_count)


@torch.no_grad()
@measured_epoch
def eval_epoch(logger, loader, model, split='val', collect_predictions=False):
    model.eval()
    time_start = time.time()
    pred_batches = [] if collect_predictions else None
    true_batches = [] if collect_predictions else None
    pred_pos_batches = [] if (collect_predictions and _is_pcqm4m_contact_dataset()) else None
    pred_neg_batches = [] if (collect_predictions and _is_pcqm4m_contact_dataset()) else None
    for batch in loader:
        if collect_predictions and _is_pcqm4m_contact_dataset():
            pred_pos_batch, pred_neg_batch = _collect_contact_predictions(batch, model)
            pred_pos_batches.extend(pred_pos_batch)
            pred_neg_batches.extend(pred_neg_batch)

        loss, _true, _pred, extra_stats = _forward_eval_batch(
            loader, model, batch, split
        )

        # 防守：检查评估中的 NaN/Inf
        if not torch.isfinite(loss):
            logging.warning(f"Non-finite loss in {split} evaluation: {loss.item()}")
        if not torch.isfinite(_pred).all():
            num_bad = (~torch.isfinite(_pred)).sum().item()
            logging.warning(f"Non-finite predictions in {split} evaluation: {num_bad}/{_pred.numel()} elements.")

        if collect_predictions:
            if not _is_pcqm4m_contact_dataset():
                pred_batches.append(_detach_to_cpu(_pred))
                true_batches.append(_detach_to_cpu(_true))
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()
    if collect_predictions:
        if _is_pcqm4m_contact_dataset():
            return {
                'pred_pos': pred_pos_batches,
                'pred_neg': pred_neg_batches,
            }
        return {
            'predictions': _merge_prediction_batches(pred_batches),
            'targets': _merge_prediction_batches(true_batches),
        }
    return None


@torch.no_grad()
def collect_split_predictions(loader, model, split='val'):
    model.eval()
    pred_batches = []
    true_batches = []
    pred_pos_batches = [] if _is_pcqm4m_contact_dataset() else None
    pred_neg_batches = [] if _is_pcqm4m_contact_dataset() else None

    for batch in loader:
        if _is_pcqm4m_contact_dataset():
            pred_pos_batch, pred_neg_batch = _collect_contact_predictions(batch, model)
            pred_pos_batches.extend(pred_pos_batch)
            pred_neg_batches.extend(pred_neg_batch)
            continue

        _, true_cpu, pred_cpu, _ = _forward_eval_batch(loader, model, batch, split)
        pred_batches.append(_detach_to_cpu(pred_cpu))
        true_batches.append(_detach_to_cpu(true_cpu))

    if _is_pcqm4m_contact_dataset():
        return {
            'pred_pos': pred_pos_batches,
            'pred_neg': pred_neg_batches,
        }
    return {
        'predictions': _merge_prediction_batches(pred_batches),
        'targets': _merge_prediction_batches(true_batches),
    }


def save_prediction_artifact(loaders, model, cur_epoch, best_epoch,
                             metric_name=None, metric_value=None,
                             split_artifacts=None):
    artifact = {
        'dataset': cfg.dataset.name,
        'run_id': cfg.run_id,
        'epoch': cur_epoch,
        'best_epoch': best_epoch,
    }
    if metric_name is not None:
        artifact['metric_name'] = metric_name
    if metric_value is not None:
        artifact['metric_value'] = metric_value

    if split_artifacts is not None:
        if 'val' not in split_artifacts or 'test' not in split_artifacts:
            raise ValueError("split_artifacts must contain both 'val' and 'test'")
        artifact['val'] = split_artifacts['val']
        artifact['test'] = split_artifacts['test']
    else:
        artifact['val'] = collect_split_predictions(loaders[1], model, split='val')
        artifact['test'] = collect_split_predictions(loaders[2], model, split='test')

    output_path = os.path.join(cfg.run_dir, 'predictions.pt')
    torch.save(artifact, output_path)
    logging.info("Saved predictions to %s", output_path)


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation, cur_epoch=cur_epoch)
        perf[0].append(loggers[0].write_epoch(cur_epoch))
        eval_split_artifacts = {}
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                split_name = split_names[i - 1]
                split_artifact = eval_epoch(
                    loggers[i], loaders[i], model,
                    split=split_name,
                    collect_predictions=getattr(cfg.train, 'save_predictions', True)
                )
                if split_artifact is not None:
                    eval_split_artifacts[split_name] = split_artifact
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if getattr(cfg.train, 'save_predictions', True):
                    metric_name = None if cfg.metric_best == 'auto' else cfg.metric_best
                    metric_value = None
                    if metric_name is not None and metric_name in perf[1][best_epoch]:
                        metric_value = perf[1][best_epoch][metric_name]
                    save_prediction_artifact(
                        loaders, model, cur_epoch, best_epoch,
                        metric_name=metric_name, metric_value=metric_value,
                        split_artifacts=eval_split_artifacts if eval_split_artifacts else None
                    )
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))
    val_perf = perf[1]

    best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                             cfg.metric_agg)()
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()
