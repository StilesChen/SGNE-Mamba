# Controlled serialization ablations (Peptides-struct)

All seven configurations use the same backbone and training settings, full edge-state transfer, and maximum segment length 512.

| Configuration | Purpose |
| --- | --- |
| a0_dfs_native | Original structural DFS reference |
| a1_dfs_segment_shuffle | Same DFS segments, independently shuffled within each segment |
| a2_dfs_fixed | Same per-graph DFS edge order, fixed-length chunks |
| a3_bfs_fixed | BFS-derived order, fixed-length chunks |
| a4_random_fixed | Graph-local random order, fixed-length chunks |
| b1_dfs_no_refinement | Native DFS, zero neighborhood signature refinement rounds |
| b2_dfs_topology_only | Native DFS, two rounds, no raw attributes in ordering |

Run from the repository root, replacing the configuration filename as needed:

```sh
python main.py --cfg configs/Mamba/struct/serialization_ablation/a0_dfs_native.yaml --repeat 5
```

Seed 0 with --repeat 5 runs seeds 0 through 4. Outputs are separated by configuration under results/serialization_ablation. Train each condition independently and select checkpoints using validation MAE.

The topology-only condition retains raw attributes in the feature encoders. Zero refinement still uses raw node attributes, degrees, and edge attributes in ordering. Fixed chunks never cross graph boundaries. Segment shuffle uses a separate CPU RNG and the sampled metadata is reused across layers. Evaluation retains the existing seed policy: with the same seed and batch, edge ordering repeats; node-order sampling follows the existing model policy.

Original random ordering should match a4 at the same seed and chunk length. Reuse historical results only after checking code, configuration and training protocol. Fixed graph priorities/tie-breaking ablation is not included.

Validation:

```sh
python tests/test_serialization_ablation.py
python tests/test_structural_dfs.py
```
