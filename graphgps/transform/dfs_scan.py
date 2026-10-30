"""Shared deterministic DFS traversal and per-graph preprocessing metadata."""
import torch


def dfs_edge_paths(edge_index, node_batch, num_nodes, max_path_len=512):
    """
    Return:
        edge_order: [num_edges], DFS-local edge order.
        path_id:    [num_edges], path id for each edge in edge_order.
    """
    edge_index_cpu = edge_index.detach().cpu()
    node_batch_cpu = node_batch.detach().cpu()

    src = edge_index_cpu[0].tolist()
    dst = edge_index_cpu[1].tolist()
    num_edges = edge_index_cpu.size(1)
    graph_ids = node_batch_cpu.tolist()

    adjacency = [[] for _ in range(num_nodes)]
    for eid, (u, v) in enumerate(zip(src, dst)):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adjacency[u].append((v, eid))

    edge_order = []
    path_ids = []
    used_edges = set()
    path_counter = 0

    unique_graphs = torch.unique(node_batch_cpu).tolist()

    for gid in unique_graphs:
        nodes = torch.where(node_batch_cpu == gid)[0].tolist()
        node_set = set(nodes)
        visited_nodes = set()

        for start in nodes:
            if start in visited_nodes:
                continue

            stack = [(start, None)]

            current_path = []

            while stack:
                cur, in_eid = stack.pop()

                if cur in visited_nodes:
                    continue

                visited_nodes.add(cur)

                if in_eid is not None and in_eid not in used_edges:
                    current_path.append(in_eid)
                    used_edges.add(in_eid)

                    # 控制单条 DFS path 的最大长度，避免过长序列
                    if len(current_path) >= max_path_len:
                        edge_order.extend(current_path)
                        path_ids.extend([path_counter] * len(current_path))
                        path_counter += 1
                        current_path = []

                for nxt, eid in reversed(adjacency[cur]):
                    if eid not in used_edges and nxt in node_set:
                        stack.append((nxt, eid))

            # 一个 DFS root 结束后，形成一个 path
            if len(current_path) > 0:
                edge_order.extend(current_path)
                path_ids.extend([path_counter] * len(current_path))
                path_counter += 1

    # 补充 DFS 未覆盖的边，例如回边、重复方向边、孤立结构边
    phases = [0] * len(edge_order)
    remaining_by_graph = {}
    for eid in range(num_edges):
        if eid not in used_edges:
            gid = graph_ids[src[eid]]
            remaining_by_graph.setdefault(gid, []).append(eid)
    for remaining in remaining_by_graph.values():
        for i in range(0, len(remaining), max_path_len):
            chunk = remaining[i:i + max_path_len]
            edge_order.extend(chunk)
            phases.extend([1] * len(chunk))
            path_ids.extend([path_counter] * len(chunk))
            path_counter += 1

    edge_order = torch.tensor(edge_order, device=edge_index.device, dtype=torch.long)
    path_ids = torch.tensor(path_ids, device=edge_index.device, dtype=torch.long)

    return edge_order, path_ids, torch.tensor(phases, device=edge_index.device, dtype=torch.long)


def add_dfs_scan_paths(data, max_path_len=512):
    """Per-graph metadata; PyG offsets the saved edge_index during batching."""
    edges = data.edge_index
    n = data.num_nodes
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
        raise ValueError("Cannot precompute DFS on invalid edge indices")
    order, paths, phases = dfs_edge_paths(
        edges, torch.zeros(n, dtype=torch.long, device=edges.device), n, max_path_len)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=edges.device)
    local_path = torch.empty_like(paths)
    local_path[order] = paths
    phase = torch.empty_like(phases)
    phase[order] = phases
    data.dfs_scan_rank = rank
    data.dfs_scan_path = local_path
    data.dfs_scan_phase = phase
    data.dfs_scan_edge_index = edges.clone()
    data.dfs_scan_node_count = torch.tensor([n], device=edges.device)
    data.dfs_scan_limit = torch.tensor([max_path_len], device=edges.device)
    return data


def assemble_dfs_scan_paths(batch, edge_index, node_batch, num_nodes, max_path_len):
    """Return None for missing/stale metadata, preserving the runtime fallback."""
    names = ('dfs_scan_rank', 'dfs_scan_path', 'dfs_scan_phase',
             'dfs_scan_edge_index', 'dfs_scan_node_count', 'dfs_scan_limit')
    if any(not torch.is_tensor(getattr(batch, name, None)) for name in names):
        return None
    saved = batch.dfs_scan_edge_index
    if saved.shape != edge_index.shape or not torch.equal(saved, edge_index):
        return None
    ptr = getattr(batch, 'ptr', None)
    counts = batch.dfs_scan_node_count.view(-1)
    if ptr is None or counts.numel() != ptr.numel() - 1:
        return None
    if not torch.equal(ptr[1:] - ptr[:-1], counts):
        return None
    expected_batch = torch.repeat_interleave(
        torch.arange(counts.numel(), device=node_batch.device), counts)
    if node_batch.numel() != num_nodes or not torch.equal(node_batch, expected_batch):
        return None
    if not bool((batch.dfs_scan_limit == max_path_len).all()):
        return None
    e = edge_index.size(1)
    rank, local_path, phase = (getattr(batch, name).view(-1)
                               for name in names[:3])
    if any(value.numel() != e for value in (rank, local_path, phase)):
        return None
    if e == 0:
        return rank, local_path
    graph = node_batch[edge_index[0]]
    # All graphs' tree paths first, then all graphs' remaining-edge chunks.
    # Local ranks are unique per graph, so keys are unique in each phase.
    graph_count = counts.numel()
    key = (phase * graph_count + graph) * (e + 1) + rank
    order = torch.argsort(key)
    ordered_graph, ordered_path = graph[order], local_path[order]
    starts = torch.ones(e, dtype=torch.bool, device=edge_index.device)
    starts[1:] = ((ordered_graph[1:] != ordered_graph[:-1])
                  | (ordered_path[1:] != ordered_path[:-1])
                  | (phase[order][1:] != phase[order][:-1]))
    return order, starts.long().cumsum(0) - 1
