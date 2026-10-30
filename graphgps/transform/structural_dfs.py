"""Scheme A: structural priorities with random, ID-independent tie breaking.

Raw attributes only; no learned features or positional eigenvectors are sorted.
One traversal is sampled per forward and shared by all edge-scan layers.
"""
from collections import deque

import torch


def build_structural_dfs(edge_index, node_batch, node_attr, edge_attr,
                         rounds=2, max_path_len=512, use_attributes=True):
    """Build reusable raw-attribute priority groups, without sampling an order.

    Historical DFS names are retained for cache/pickle and caller compatibility;
    these groups also support BFS and graph-local random edge permutations.
    """
    if rounds < 0 or max_path_len < 1:
        raise ValueError('rounds >= 0 and max_path_len >= 1 required')
    n, e = node_batch.numel(), edge_index.size(1)
    edges = edge_index.detach().cpu()
    groups = node_batch.detach().cpu().tolist()
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
        raise ValueError('Invalid structural scan edge indices')

    def rows(value, count):
        if value is None:
            return [()] * count
        if value.size(0) != count or not bool(torch.isfinite(value).all()):
            raise ValueError('Raw structural attributes must be finite and aligned')
        if count == 0:
            return []
        return [tuple(row) for row in value.detach().cpu().reshape(count, -1).tolist()]

    x, a = (rows(node_attr, n), rows(edge_attr, e)) if use_attributes else ([()] * n, [()] * e)
    src, dst = edges.tolist()
    adjacency = [[] for _ in range(n)]
    incoming = [[] for _ in range(n)]
    graph_nodes = {}
    for v, gid in enumerate(groups):
        graph_nodes.setdefault(gid, []).append(v)
    for eid, (u, v) in enumerate(zip(src, dst)):
        if groups[u] != groups[v]:
            raise ValueError('Structural scan does not accept cross-graph edges')
        adjacency[u].append((v, eid))
        incoming[v].append((u, eid))

    def compress(signatures):
        # Sorted tuple vocabulary: no randomized Python hashes or input-order IDs.
        vocab = {key: i for i, key in enumerate(sorted(set(signatures)))}
        return [vocab[key] for key in signatures]

    colors = compress([(x[v], len(adjacency[v]), len(incoming[v])) for v in range(n)])
    for _ in range(rounds):
        colors = compress([(colors[v],
                            tuple(sorted((a[eid], colors[u]) for u, eid in adjacency[v])),
                            tuple(sorted((a[eid], colors[u]) for u, eid in incoming[v])))
                           for v in range(n)])

    def partition(items, key):
        groups = {}
        for item in items:
            groups.setdefault(key(item), []).append(item)
        return tuple(tuple(groups[k]) for k in sorted(groups))

    return dict(n=n, e=e, src=src, dst=dst, groups=groups,
        roots={gid: partition(nodes, lambda v: colors[v])
               for gid, nodes in graph_nodes.items()},
        neighbors=[partition(neighbors, lambda pair: (a[pair[1]], colors[pair[0]]))
                   for neighbors in adjacency],
        remaining={gid: partition([eid for v in nodes for _, eid in adjacency[v]],
                       lambda eid: (colors[src[eid]], a[eid], colors[dst[eid]]))
                   for gid, nodes in graph_nodes.items()})


def structural_dfs_paths(edge_index, node_batch, node_attr, edge_attr,
                         rounds=2, max_path_len=512, generator=None,
                         node_priority=None, edge_priority=None, structure=None,
                         scan_order='dfs', use_attributes=True):
    """Return edge order, sequence IDs and tree/remaining phases.

    DFS/BFS use structural groups with random tie breaking. Random mode sorts
    all edges per graph by uniform random priorities, ignoring these groups.
    Explicit priorities support coupled relabeling diagnostics; equal seeds
    alone do not guarantee identical scans after relabeling symmetric graphs.
    """
    if scan_order not in ('dfs', 'bfs', 'random'):
        raise ValueError(f'Unsupported structural scan order: {scan_order}')
    if max_path_len < 1:
        raise ValueError('max_path_len must be positive')
    structure = structure if structure is not None else build_structural_dfs(
        edge_index, node_batch, node_attr, edge_attr, rounds, max_path_len, use_attributes)
    n, e = structure['n'], structure['e']
    src, dst = structure['src'], structure['dst']

    def priorities(value, count):
        if value is None:
            return torch.randperm(count, generator=generator).tolist()
        value = torch.as_tensor(value).cpu().view(-1)
        if not torch.equal(value.sort().values, torch.arange(count)):
            raise ValueError('Tie priorities must be a permutation of 0..count-1')
        return value.tolist()

    np, ep = priorities(node_priority, n), priorities(edge_priority, e)
    return _sample_structural_paths(structure, np, ep, max_path_len,
                                    scan_order, edge_index.device)


def _sample_structural_paths(structure, np, ep, max_path_len, scan_order, device):
    """Internal sampler accepting unique priorities, not necessarily local ranks.

    Callers own validation. Only relative priority order is used below.
    """
    src, dst = structure['src'], structure['dst']
    order, paths, phases = [], [], []
    path_count = 0

    def append_path(sequence, phase):
        nonlocal path_count
        for start in range(0, len(sequence), max_path_len):
            chunk = sequence[start:start + max_path_len]
            order.extend(chunk)
            paths.extend([path_count] * len(chunk))
            phases.extend([phase] * len(chunk))
            path_count += 1

    if scan_order == 'random':
        # Uniform graph-local edge permutation, independent of structural colors.
        # Keep the same maximum sequence length and graph isolation as DFS.
        for gid in sorted(structure['remaining']):
            edge_ids = [eid for group in structure['remaining'][gid] for eid in group]
            append_path(sorted(edge_ids, key=lambda eid: ep[eid]), 0)
        return tuple(torch.tensor(value, dtype=torch.long, device=device)
                     for value in (order, paths, phases))

    # Dense local IDs permit constant-time flags without Python hash tables.
    # Singleton structural groups already have their final ordering.
    visited, used = bytearray(structure['n']), bytearray(structure['e'])
    adjacency = [[pair for group in groups
                  for pair in (group if len(group) < 2 else sorted(
                      group, key=lambda pair: (np[pair[0]], ep[pair[1]])))]
                 for groups in structure['neighbors']]
    for gid in sorted(structure['roots']):
        roots = [v for group in structure['roots'][gid]
                 for v in (group if len(group) < 2 else sorted(group, key=np.__getitem__))]
        for root in roots:
            if visited[root]:
                continue
            if scan_order == 'bfs':
                queue, tree = deque([root]), []
                visited[root] = 1
                while queue:
                    v = queue.popleft()
                    for u, eid in adjacency[v]:
                        if not visited[u]:
                            visited[u] = 1  # Discover once, including parallel edges.
                            queue.append(u)
                            tree.append(eid)
                            used[eid] = 1
                append_path(tree, 0)
                continue
            stack, tree = [(root, None)], []
            while stack:
                v, eid = stack.pop()
                if visited[v]:
                    continue
                visited[v] = 1
                if eid is not None:
                    tree.append(eid)
                    used[eid] = 1
                for u, next_edge in reversed(adjacency[v]):
                    if not visited[u]:
                        stack.append((u, next_edge))
            append_path(tree, 0)

    for gid in sorted(structure['remaining']):
        remaining = [eid for group in structure['remaining'][gid]
                     for eid in sorted((eid for eid in group if not used[eid]),
                                       key=lambda eid: (np[src[eid]], np[dst[eid]], ep[eid]))]
        append_path(remaining, 1)
    return tuple(torch.tensor(value, dtype=torch.long, device=device)
                 for value in (order, paths, phases))


class StructuralDFSCache:
    """CPU-only immutable-by-convention graph metadata; PyG collates a list.

No .to() method: PyG leaves this object on CPU when moving the batch.
Snapshots detect transforms that change topology, raw features or numbering.
"""
    def __init__(self, edges, x, attr, rounds, use_attributes=True):
        self.edges = edges.detach().cpu().clone()
        self.x = x.detach().cpu().clone()
        self.attr = None if attr is None else attr.detach().cpu().clone()
        self.rounds = rounds
        self.use_attributes = use_attributes
        self.structure = build_structural_dfs(self.edges, torch.zeros(len(x), dtype=torch.long),
                                              self.x, self.attr, rounds, use_attributes=use_attributes)

    def matches(self, edges, x, attr, rounds, use_attributes=True):
        return (getattr(self, 'use_attributes', True) == use_attributes
                and self.rounds == rounds and torch.equal(self.edges, edges)
                and torch.equal(self.x, x)
                and ((self.attr is None and attr is None) or
                     (self.attr is not None and attr is not None and torch.equal(self.attr, attr))))

    def __deepcopy__(self, memo):
        return self  # sampling never mutates these groups or snapshots


def add_structural_dfs_cache(data, rounds=2, use_attributes=True):
    data.dfs_structural_cache = StructuralDFSCache(
        data.edge_index, data.x, getattr(data, 'edge_attr', None), rounds, use_attributes)
    return data


def _graph_indices(groups):
    """Group once, preserving original order; use views for ordered PyG batches.

    Derive boundaries from actual membership, not potentially stale PyG slices.
    The stable-sort fallback supports reordered nodes/edges and sparse graph IDs.
    """
    ids, counts = torch.unique_consecutive(groups, return_counts=True)
    ordered = ids.numel() < 2 or bool((ids[1:] > ids[:-1]).all())
    permutation = None
    if not ordered:
        permutation = torch.argsort(groups, stable=True)
        ids, counts = torch.unique_consecutive(groups[permutation], return_counts=True)
    result, start = {}, 0
    for gid, count in zip(ids.tolist(), counts.tolist()):
        end = start + count
        result[gid] = slice(start, end) if ordered else permutation[start:end]
        start = end
    return result


def cached_structural_dfs_paths(batch, rounds, generator, scan_order='dfs',
                                max_path_len=512, use_attributes=True):
    """Assemble graph-local caches with the same global random priorities."""
    if scan_order not in ('dfs', 'bfs', 'random'):
        raise ValueError(f'Unsupported structural scan order: {scan_order}')
    if max_path_len < 1:
        raise ValueError('max_path_len must be positive')
    if batch.edge_index.device.type != 'cpu':
        # External callers that do not use prepare_model_batch retain correctness.
        return structural_dfs_paths(batch.edge_index, batch.batch, batch.x,
                                    getattr(batch, 'edge_attr', None), rounds=rounds,
                                    generator=generator, scan_order=scan_order,
                                    max_path_len=max_path_len, use_attributes=use_attributes)
    caches = getattr(batch, 'dfs_structural_cache', None)
    if not isinstance(caches, (list, tuple)):
        caches = [caches]
    n, e = batch.batch.numel(), batch.edge_index.size(1)
    if e:
        if int(batch.edge_index.min()) < 0 or int(batch.edge_index.max()) >= n:
            raise ValueError('Invalid structural scan edge indices')
    edge_graphs = batch.batch[batch.edge_index[0]]
    if e:
        if not torch.equal(edge_graphs, batch.batch[batch.edge_index[1]]):
            raise ValueError('Structural scan does not accept cross-graph edges')
    np = torch.randperm(n, generator=generator)
    ep = torch.randperm(e, generator=generator)
    results = []
    attr = getattr(batch, 'edge_attr', None)
    node_groups = _graph_indices(batch.batch)
    edge_groups = _graph_indices(edge_graphs)
    inverse = None
    for i, (gid, nodes) in enumerate(node_groups.items()):
        edge_ids = edge_groups.get(gid, slice(0, 0))
        if isinstance(nodes, slice):
            edges = batch.edge_index[:, edge_ids] - nodes.start
        else:
            if inverse is None:
                inverse = torch.empty(n, dtype=torch.long)
            inverse[nodes] = torch.arange(nodes.numel())
            edges = inverse[batch.edge_index[:, edge_ids]]
        x = batch.x[nodes]
        a = None if attr is None else attr[edge_ids]
        cache = caches[i] if i < len(caches) else None
        if not isinstance(cache, StructuralDFSCache) or not cache.matches(edges, x, a, rounds, use_attributes):
            cache = StructuralDFSCache(edges, x, a, rounds, use_attributes)
        # Slices of randperm contain unique priorities. Ranking them locally
        # would preserve every comparison but require four redundant sorts.
        order, paths, phases = _sample_structural_paths(
            cache.structure, np[nodes].tolist(), ep[edge_ids].tolist(),
            max_path_len, scan_order, edges.device)
        global_order = order + edge_ids.start if isinstance(edge_ids, slice) else edge_ids[order]
        results.append((global_order, paths, phases))
    orders, paths_out, phases_out = [], [], []
    counter = 0
    for phase in (0, 1):
        for order, paths, phases in results:
            selected = phases == phase
            if not bool(selected.any()):
                continue
            local = paths[selected]
            orders.append(order[selected])
            paths_out.append(local - local[0] + counter)
            phases_out.append(phases[selected])
            counter += int(local[-1] - local[0]) + 1
    empty = torch.empty(0, dtype=torch.long)
    return tuple(torch.cat(v) if v else empty.clone() for v in (orders, paths_out, phases_out))


def transform_edge_segments(order, paths, edge_index, node_batch,
                            segmentation='native', max_path_len=512,
                            within_segment_shuffle=False, shuffle_generator=None):
    """Postprocess after phase merging; never merge edges across graphs.

    Shuffle uses a dedicated CPU generator and never consumes the global RNG.
    """
    if segmentation not in ('native', 'fixed_chunks') or max_path_len < 1:
        raise ValueError('Expected native/fixed_chunks and positive segment length')
    if segmentation == 'fixed_chunks' and order.numel():
        gids = node_batch[edge_index[0, order]]
        orders, segments = [], []
        offset = 0
        for gid in gids.unique(sorted=True):
            selected = order[gids == gid]
            ids = torch.arange(selected.numel(), device=order.device) // max_path_len
            orders.append(selected)
            segments.append(ids + offset)
            offset += int(ids[-1]) + 1
        order, paths = torch.cat(orders), torch.cat(segments)
    if within_segment_shuffle:
        if shuffle_generator is None:
            raise ValueError('Segment shuffle requires an independent generator')
        order = order.clone()
        for pid in paths.unique_consecutive():
            positions = (paths == pid).nonzero().view(-1)
            permutation = torch.randperm(positions.numel(), generator=shuffle_generator).to(order.device)
            order[positions] = order[positions[permutation]]
    return order, paths


def prepare_structural_dfs(batch, rounds, generator, scan_order='dfs',
                           segmentation='native', max_path_len=512,
                           within_segment_shuffle=False, use_attributes=True,
                           shuffle_generator=None):
    """Overwrite per-forward metadata before encoders mutate x/edge_attr."""
    order, paths, _ = cached_structural_dfs_paths(
        batch, rounds, generator, scan_order=scan_order,
        max_path_len=max_path_len, use_attributes=use_attributes)
    order, paths = transform_edge_segments(
        order, paths, batch.edge_index, batch.batch, segmentation, max_path_len,
        within_segment_shuffle, shuffle_generator)
    batch._gps_structural_dfs = (batch.edge_index.clone(), batch.batch.clone(), order, paths)


def prepare_model_batch(model, batch):
    """Prepare on CPU before batch.to(device), supporting GraphGym wrappers."""
    target = model
    while not hasattr(target, 'prepare_structural_batch'):
        next_target = getattr(target, 'model', getattr(target, 'module', None))
        if next_target is None or next_target is target:
            return
        target = next_target
    target.prepare_structural_batch(batch)


def activate_structural_dfs(batch):
    """Consume the public path tensors after PyG moves them to the device."""
    if getattr(batch, '_gps_structural_prepared', False):
        names = ('edges', 'batch', 'order', 'paths')
        batch._gps_structural_dfs = tuple(getattr(batch, 'dfs_runtime_' + name) for name in names)
        for name in names:
            delattr(batch, 'dfs_runtime_' + name)
        batch._gps_structural_prepared = False


def structural_dfs_seed(seed, training, step):
    """Independent CPU stream: never consume the degree scanner's RNG stream."""
    return (int(seed) + 1000003 * (int(step) + 1 if training else 0)) % (2**63 - 1)
