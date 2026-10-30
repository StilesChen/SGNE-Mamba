"""Disk cache for explicitly deterministic, graph-local preprocessing only."""
import hashlib
import inspect
import json
import logging
import os
import pickle
import tempfile
from importlib import import_module

import torch


class DeterministicTransformCache:
    """Cache output fields, never whole graphs, labels, splits or RNG state.

    Tensor contents (including edge ordering) form the key. Source and library
    versions invalidate old results. Atomic replacement supports concurrent runs.
    Only use with deterministic transforms that do not consume global RNG.
    """
    def __init__(self, transform, root, namespace, settings, fields=(), structural_type=None):
        self.transform = transform
        self.root = os.path.join(root, namespace)
        self.fields = tuple(fields)
        self.structural_type = structural_type
        self.hits = self.misses = 0
        function = getattr(transform, 'func', transform)
        source = inspect.getsourcefile(function)
        with open(source, 'rb') as stream:
            source_hash = hashlib.sha256(stream.read()).hexdigest()
        libraries = {name: str(import_module(name).__version__)
                     for name in ('torch', 'numpy', 'scipy', 'torch_geometric')}
        self.prefix = json.dumps(dict(schema=1, namespace=namespace, settings=settings,
                                      source=source_hash, libraries=libraries),
                                 sort_keys=True).encode('utf-8')

    def _key(self, data):
        digest = hashlib.sha256(self.prefix)
        digest.update(str(int(data.num_nodes)).encode('ascii'))
        for name in ('edge_index', 'x', 'edge_attr'):
            tensor = getattr(data, name, None)
            digest.update(name.encode('ascii'))
            if tensor is None:
                digest.update(b'none')
                continue
            value = tensor.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), value.dtype)).encode('ascii'))
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _restore(self, data, payload):
        if self.structural_type is not None:
            cache = self.structural_type.__new__(self.structural_type)
            cache.edges = data.edge_index.detach().cpu().clone()
            cache.x = data.x.detach().cpu().clone()
            attr = getattr(data, 'edge_attr', None)
            cache.attr = None if attr is None else attr.detach().cpu().clone()
            cache.rounds = payload['rounds']
            cache.use_attributes = payload['use_attributes']
            cache.structure = payload['structure']
            data.dfs_structural_cache = cache
        else:
            if set(payload) != set(self.fields):
                raise ValueError('Unexpected preprocessing cache fields')
            for name in self.fields:
                if not torch.is_tensor(payload[name]):
                    raise ValueError('Preprocessing cache must contain tensors')
            for name in self.fields:
                setattr(data, name, payload[name])
        return data

    def __call__(self, data):
        key = self._key(data)
        directory = os.path.join(self.root, key[:2])
        path = os.path.join(directory, key + '.pt')
        try:
            entry = torch.load(path, map_location='cpu', weights_only=True)
            if entry['key'] != key:
                raise ValueError('Preprocessing cache key mismatch')
            result = self._restore(data, entry['payload'])
            self.hits += 1
            return result
        except (OSError, EOFError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
            # Missing/corrupt cache is optional: recompute from current input.
            pass
        self.misses += 1
        result = self.transform(data)
        if self.structural_type is not None:
            cache = result.dfs_structural_cache
            payload = dict(rounds=cache.rounds, use_attributes=cache.use_attributes,
                           structure=cache.structure)
        else:
            payload = {name: getattr(result, name).detach().cpu() for name in self.fields}
        temporary = None
        try:
            os.makedirs(directory, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix='pending-', suffix='.pt', dir=directory)
            os.close(descriptor)
            torch.save(dict(key=key, payload=payload), temporary)
            os.replace(temporary, path)
        except OSError as error:
            logging.warning('Unable to write preprocessing cache %s: %s', path, error)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return result

    def report(self):
        logging.info('Deterministic preprocessing cache %s: hits=%d misses=%d',
                     self.root, self.hits, self.misses)
