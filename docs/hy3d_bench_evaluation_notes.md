# HY3D-Bench Evaluation Notes For ICPA Cotton 3D

## What HY3D-Bench Adds

HY3D-Bench is a 2026 Tencent Hunyuan3D benchmark for 3D asset generation. It is
not an agricultural phenotyping benchmark, but it is useful because it shows how
modern 3D work is being evaluated at scale:

- full-object assets are cleaned, normalized, watertight, and paired with
  standardized multi-view renderings and sampled points;
- part-level decompositions are provided for fine-grained structural evaluation;
- synthetic assets are used to rebalance long-tail categories;
- a compact Hunyuan3D baseline is trained to show reproducibility.

For our ICPA paper, HY3D-Bench should be cited only as a general 3D benchmark
reference, not as a direct cotton baseline.

## Transferable Ideas

| HY3D-Bench idea | Cotton 3D translation |
|---|---|
| Watertight, cleaned meshes | Report geometry validity for any mesh output: holes, connected components, non-manifold edges if available. |
| Multi-view renderings | Hold out UAV frames and evaluate novel-view or reprojected-mask consistency. |
| Sampled points | Standardize point-cloud sampling before Chamfer/F-score comparisons. |
| Part-level decomposition | Treat cotton organs as parts: boll, canopy, stem/branch, soil/background. |
| Synthetic long-tail data | Use synthetic/augmented cotton bolls only for robustness or ablation, never as the main evidence. |
| Baseline model release | Keep a simple reproducible baseline: detector + monocular depth proxy + PLY export. |

## Evaluation Protocol We Should Add

The paper should separate evaluation into four levels:

1. **Image evidence.** Registered image count, blur/sharpness, overlap, phase split,
   and detector count.
2. **Geometry evidence.** Point density, connected components, Chamfer/F-score where
   reference geometry exists, reprojection error, and held-out view consistency.
3. **Organ evidence.** Boll instance recovery, duplicate reduction, view support,
   visibility score, diameter/volume proxy uncertainty, and high-confidence coverage.
4. **Reporting evidence.** LLM/MoE schema validity, faithfulness to measured JSON,
   unsupported-claim rate, latency, and expert alignment.

## Paper Wording

Use this carefully:

> Inspired by recent large-scale 3D benchmark design, we report reconstruction
> quality at the level of standardized sampled points, multi-view consistency,
> and organ-level part structure rather than relying only on visual inspection.
> In the cotton setting, the relevant parts are not generic object components
> but agronomic organs: visible bolls, canopy/leaf mass, branches, and soil.

Avoid saying:

> We use HY3D-Bench for cotton evaluation.

That would be inaccurate unless we actually download and run their dataset or
baseline, which is unnecessary for this ICPA paper.

## Sources

- HY3D-Bench paper page: https://huggingface.co/papers/2602.03907
- HY3D-Bench GitHub: https://github.com/Tencent-Hunyuan/HY3D-Bench
- HY3D-Bench dataset card: https://huggingface.co/datasets/tencent/HY3D-Bench
