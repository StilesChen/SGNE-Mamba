"""Measure real models and emit parameter-matched single-scan configs.

Run in the project's training environment. Loads the dataset once for GraphGym
metadata; does not train, evaluate, or select widths using prediction metrics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def choose_candidate(rows, target, tolerance, scan=None):
    eligible = [r for r in rows if abs(r['parameters'] / target - 1) <= tolerance]
    if not eligible:
        nearest = sorted(rows, key=lambda r: (abs(r['parameters'] - target), r['width']))[:5]
        label = f"{scan}-only" if scan else 'candidate'
        details = '\n'.join(
            f"  width={r['width']}: parameters={r['parameters']:,}, "
            f"relative_difference={r['parameters'] / target - 1:+.3%}"
            for r in nearest
        )
        best = abs(nearest[0]['parameters'] / target - 1)
        raise ValueError(
            f'No {label} width satisfies tolerance ±{tolerance:.3%} '
            f'(target={target:,} parameters). Closest candidates:\n{details}\n'
            f"Smallest achievable mismatch in this width grid: {best:.3%}. "
            'Increase --tolerance to at least this value (with a small margin), '
            'or choose a wider/different model configuration.'
        )
    # Prefer a slightly larger control to avoid under-budgeting the single scan.
    above = [r for r in eligible if r['parameters'] >= target]
    return min(above or eligible, key=lambda r: (abs(r['parameters'] - target), r['width']))


def prepare_audit_batch(model, data):
    """Support both bare GPSModel and GraphGymModule-returning PyG versions."""
    backbone = getattr(model, 'model', model)
    prepare = getattr(backbone, 'prepare_structural_batch', None)
    if callable(prepare):
        prepare(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', default='configs/Mamba/struct/struct_mamba_edge_mfsg.yaml')
    parser.add_argument('--output-dir', default='configs/Mamba/struct/capacity_ablation/generated')
    parser.add_argument('--min-width', type=int, default=96)
    parser.add_argument('--max-width', type=int, default=192)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--tolerance', type=float, default=0.01,
                        help='Maximum relative parameter mismatch (default: 0.01).')
    parser.add_argument('--dataset-dir', help='Optional existing dataset directory.')
    parser.add_argument('--check-device', help='Optional real forward/backward audit, e.g. cuda:0.')
    args = parser.parse_args()
    if args.step < 1 or args.min_width < 1 or args.max_width < args.min_width:
        parser.error('Invalid width grid')
    if not 0 < args.tolerance < 1:
        parser.error('tolerance must be between 0 and 1')
    destination = Path(args.output_dir)
    if destination.exists():
        parser.error('Output directory already exists; use a fresh directory to preserve prior selections.')

    import torch
    import yaml
    import graphgps  # noqa: F401 -- same registration as main.py
    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.model_builder import create_model

    set_cfg(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=args.cfg, opts=[]))
    if (cfg.gt.layer_type != 'CustomGatedGCN+Mamba_Edge'
            or cfg.gt.scan_target != 'both' or cfg.gt.edge_node_fusion_mode != 'mfsg'):
        parser.error('Use the full CustomGatedGCN+Mamba_Edge / both / mfsg configuration.')
    if cfg.pretrained.dir:
        parser.error('Pretrained models are not supported by this experiment.')
    if args.dataset_dir:
        cfg.dataset.dir = args.dataset_dir
    cfg.accelerator = 'cpu'
    torch.set_num_threads(cfg.num_threads)
    loaders = create_loader()  # Obtain the same input/output metadata as training.
    base = cfg.clone()

    def measure(scan, width):
        cfg.merge_from_other_cfg(base)
        cfg.gt.scan_target = scan
        cfg.gt.edge_node_fusion_mode = 'mfsg' if scan == 'both' else 'single'
        cfg.gt.dim_hidden = cfg.gnn.dim_inner = width
        torch.manual_seed(base.seed)
        model = create_model(to_device=False)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if total != count:
            raise ValueError('Frozen parameters found; define the capacity budget explicitly.')
        del model
        return dict(scan=scan, width=width, parameters=count)

    full = measure('both', base.gt.dim_hidden)
    target = full['parameters']
    rows = [full]
    chosen = {}
    for scan in ('node', 'edge'):
        candidates = [measure(scan, w) for w in range(args.min_width, args.max_width + 1, args.step)]
        rows.extend(candidates)
        chosen[scan] = choose_candidate(candidates, target, args.tolerance, scan=scan)
        print(scan, chosen[scan], flush=True)

    if args.check_device:
        from torch_geometric.graphgym.loss import compute_loss
        sample = next(iter(loaders[0]))
        for row in (full, chosen['node'], chosen['edge']):
            measure(row['scan'], row['width'])  # Restore exactly this configuration.
            model = create_model(to_device=False).to(args.check_device).train()
            data = sample.clone()
            data.split = 'train'
            prepare_audit_batch(model, data)
            data = data.to(args.check_device)
            prediction, truth = model(data)
            loss, _ = compute_loss(prediction, truth)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss for {row}')
            loss.backward()
            missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            invalid = [n for n, p in model.named_parameters()
                       if p.grad is not None and not torch.isfinite(p.grad).all()]
            if missing or invalid:
                raise RuntimeError(f'Gradient audit failed: missing={missing}, nonfinite={invalid}')
            print(f"Real backward audit passed: {row['scan']} width={row['width']}", flush=True)
            del model, data, prediction, truth, loss

    # Only publish runnable configs after BOTH searches pass the tolerance.
    source = Path(args.cfg).read_text(encoding='utf-8')
    destination.mkdir(parents=True)
    for scan in ('node', 'edge'):
        variant = yaml.safe_load(source)
        variant['gt']['scan_target'] = scan
        variant['gt']['edge_node_fusion_mode'] = 'single'
        variant['gt']['dim_hidden'] = chosen[scan]['width']
        variant['gnn']['dim_inner'] = chosen[scan]['width']
        variant['out_dir'] = 'results/capacity_ablation'
        if args.dataset_dir:
            variant['dataset']['dir'] = args.dataset_dir
        (destination / f'struct_{scan}_only_pm.yaml').write_text(
            yaml.safe_dump(variant, sort_keys=False), encoding='utf-8')
    for row in rows:
        row['relative_difference'] = row['parameters'] / target - 1
    with (destination / 'parameter_search.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(base_config=str(Path(args.cfg).resolve()),
                  base_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  full=full, selected=chosen, tolerance=args.tolerance,
                  selection='closest above target within tolerance, otherwise closest below',
                  dim_in=cfg.share.dim_in, dim_out=cfg.share.dim_out,
                  torch_version=torch.__version__,
                  backward_checked_device=args.check_device,
                  note='No accuracy evaluation; optional gradient audit uses one training batch.')
    (destination / 'selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Includes resolved defaults and loader-populated metadata for audit.
    (destination / 'resolved_base.yaml').write_text(base.dump(), encoding='utf-8')
    print(f'Full parameters: {target}; wrote configs and audit to {destination}')
    del loaders


if __name__ == '__main__':
    main()
