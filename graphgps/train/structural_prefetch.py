"""One-batch DFS prefetch without moving dataset access to a worker thread.

The training loop requests the next batch only AFTER backward/optimizer calls.
Dataset/sampler RNG therefore retains its order relative to model RNG. Only
pure structural preparation (with explicit private generators) runs in a
worker. The existing logger must remain RNG-free, as graphgps.logger is.
This deliberately overlaps only the update/logging tail, not a full forward.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import torch

from graphgps.transform.structural_dfs import structural_dfs_seed


class StructuralPrefetchLoader:
    def __init__(self, loader, target):
        self.loader = loader
        self.target = target
        self.iterator = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dfs-prepare')
        self.pending = None
        self.exhausted = False
        self.error = None
        self.step = None

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        return self

    def _prepare(self, batch, seed):
        # No model forward, device transfer, global RNG or step-buffer writes.
        self.target.prepare_structural_batch(batch, seed=seed)
        batch._gps_structural_prefetched = True
        return batch

    def prefetch_next(self):
        if self.pending is not None or self.exhausted or self.error is not None:
            return
        try:
            if self.iterator is None:
                self.iterator = iter(self.loader)
            # Keep lazy samplers and stochastic dataset transforms on the
            # training thread, after all current-batch stochastic computation.
            batch = next(self.iterator)
            if self.step is None:
                # Read the checkpointed counter once per epoch; forward is
                # still its sole owner and increments even on non-finite loss.
                self.step = int(self.target._structural_dfs_step.item())
            seed = structural_dfs_seed(torch.initial_seed(), True, self.step)
            batch.split = 'train'
            self.pending = self.executor.submit(self._prepare, batch, seed)
        except StopIteration:
            self.exhausted = True
        except Exception as error:
            # Report at the original iteration boundary, after current logging.
            self.error = error

    def __next__(self):
        self.prefetch_next()
        if self.error is not None:
            raise self.error
        if self.pending is None:
            raise StopIteration
        batch = self.pending.result()
        self.pending = None
        self.step += 1
        return batch

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending = None
        self.iterator = None


def with_structural_prefetch(function):
    @wraps(function)
    def wrapped(logger, loader, model, *args, **kwargs):
        from torch_geometric.graphgym.config import cfg
        target = model
        while not hasattr(target, 'prepare_structural_batch'):
            child = getattr(target, 'model', getattr(target, 'module', None))
            if child is None or child is target:
                break
            target = child
        if (not getattr(cfg.train, 'structural_dfs_prefetch', False)
                or not getattr(target, 'structural_edge_scan', False)):
            return function(logger, loader, model, *args, **kwargs)
        prepared = StructuralPrefetchLoader(loader, target)
        try:
            return function(logger, prepared, model, *args, **kwargs)
        finally:
            prepared.close()
    return wrapped
