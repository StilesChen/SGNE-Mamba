"""Opt-in timing and bounded CPU/CUDA traces for training and evaluation."""
import functools
import logging
import os
import time
from contextlib import nullcontext

import torch
from contextvars import ContextVar

_profiling = ContextVar('gps_profiling', default=False)


def performance_region(name):
    return torch.profiler.record_function(name) if _profiling.get() else nullcontext()


def measured_epoch(function):
    @functools.wraps(function)
    def wrapped(logger, loader, model, *args, **kwargs):
        from torch_geometric.graphgym.config import cfg
        timing = getattr(cfg.train, 'performance_timing', False)
        layouts = getattr(cfg.train, 'performance_layout_batches', 0)
        split = 'train' if function.__name__ == 'train_epoch' else kwargs.get('split', 'val')
        epoch = kwargs.get('cur_epoch', None)
        tracing = (split == 'train' and getattr(cfg.train, 'performance_profile', False)
                   and epoch == getattr(cfg.train, 'performance_profile_epoch', 2))
        if not (timing or layouts or tracing):
            return function(logger, loader, model, *args, **kwargs)
        device = next(model.parameters()).device
        layers = [m for m in model.modules() if hasattr(m, '_cached_dfs_edge_paths')]
        for index, layer in enumerate(layers):
            layer._perf_enabled = tracing
            layer._perf_cache_enabled = bool(timing or layouts)
            layer._perf_layout = None
            layer._perf_layout_stats = {}
            layer._perf_last_node = layer._perf_last_edge = None
            layer._dfs_cache_hits = layer._dfs_cache_misses = 0
            layer._dfs_precomputed_uses = layer._dfs_runtime_builds = 0
        wait = getattr(cfg.train, 'performance_profile_wait', 10)
        warmup = getattr(cfg.train, 'performance_profile_warmup', 3)
        active = getattr(cfg.train, 'performance_profile_active', 5)
        output_dir = os.path.join(cfg.run_dir, 'profiler', 'epoch_%s' % epoch)
        if tracing:
            if wait < 0 or warmup < 0 or active < 1:
                raise ValueError('Profiler requires wait/warmup >= 0 and active >= 1')
            os.makedirs(output_dir, exist_ok=True)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            def save_trace(profiler):
                profiler.export_chrome_trace(os.path.join(output_dir, 'trace.json'))
                with open(os.path.join(output_dir, 'operators.txt'), 'w', encoding='utf-8') as stream:
                    stream.write(profiler.key_averages().table(
                        sort_by='self_cpu_time_total', row_limit=60))
                    if device.type == 'cuda':
                        stream.write('\n' + profiler.key_averages().table(
                            sort_by='self_cuda_time_total', row_limit=60))
                logging.info('Performance trace saved to %s', output_dir)

            profiler_context = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
                on_trace_ready=save_trace, record_shapes=True,
                profile_memory=False, with_stack=False)
            if len(loader) < wait + warmup + active:
                logging.warning('Loader shorter than profiler schedule; trace may be partial.')
        else:
            profiler_context = nullcontext(None)

        class SampledLoader:
            def __len__(self):
                return len(loader)

            def __getattr__(self, name):
                return getattr(loader, name)

            def __iter__(self):
                for iteration, batch in enumerate(loader):
                    for index, layer in enumerate(layers):
                        layer._perf_layout = (split, iteration) if index == 0 and iteration < layouts else None
                    try:
                        yield batch
                    finally:
                        if profiler is not None:
                            profiler.step()

        def synchronize():
            if timing and device.type == 'cuda':
                torch.cuda.synchronize(device)

        synchronize()
        start = time.perf_counter()
        token = _profiling.set(tracing)
        try:
            with profiler_context as profiler:
                result = function(logger, SampledLoader(), model, *args, **kwargs)
            synchronize()
            if timing:
                logging.info('Performance timing split=%s seconds=%.3f profiled=%s',
                             split, time.perf_counter() - start, tracing)
            if timing or layouts:
                logging.info('DFS cache split=%s hits=%d misses=%d precomputed=%d runtime_builds=%d', split,
                             sum(m._dfs_cache_hits for m in layers),
                             sum(m._dfs_cache_misses for m in layers),
                             sum(m._dfs_precomputed_uses for m in layers),
                             sum(m._dfs_runtime_builds for m in layers))
            if layers:
                for kind, (valid, dense, samples) in layers[0]._perf_layout_stats.items():
                    logging.info('Layout total split=%s kind=%s samples=%d valid=%d dense=%d padding=%.2f%%',
                                 split, kind, samples, valid, dense,
                                 100 * (1 - valid / max(dense, 1)))
            return result
        finally:
            _profiling.reset(token)
            for layer in layers:
                layer._perf_enabled = False
                layer._perf_cache_enabled = False
                layer._perf_layout = None
    return wrapped
