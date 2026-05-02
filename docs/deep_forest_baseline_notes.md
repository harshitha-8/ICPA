# Deep-Forest Baseline Notes

## Why This Figure Matters

The deep-forest diagram in the ISPRS cotton 3D reconstruction paper is not a 3D
reconstruction method. It is a supervised tabular-learning model: multi-grained
feature scanning followed by a cascade of random forests. In that paper context,
it supports prediction from extracted cotton features.

For our project, the useful lesson is:

> After geometry and boll extraction produce structured features, a lightweight
> non-neural model can serve as a strong, interpretable baseline for candidate
> validation or plot-level trait prediction.

## Where It Fits

Do use it for:

- boll candidate validation: valid boll vs leaf/soil/noise;
- adhered or merged boll rejection;
- plot-cell trait prediction from mapped counts/diameter/volume proxies;
- yield or quality prediction if plot-level ground truth is available.

Do not use it as:

- the 3D reconstruction engine;
- the main novelty of the paper;
- a substitute for calibrated geometry.

## Feature Inputs From Our App

The current app already exports candidate-level features:

- diameter proxy;
- volume proxy;
- visibility proxy;
- depth score;
- lint fraction;
- green fraction;
- brightness score;
- size score;
- extraction quality;
- row/column cell count summaries.

These are suitable inputs to a Deep-Forest-style baseline once a small labeled
set is available.

## Evaluation Placement

Add it to the experiments as a tabular baseline:

| Task | Baselines | Metrics |
|---|---|---|
| Candidate validation | heuristic extraction score, random forest cascade, optional XGBoost/MLP | precision, recall, F1, PR-AUC |
| Adhered boll rejection | volume threshold, random forest cascade | F1, false-merge rate |
| Plot-cell trait prediction | count-only, count+volume, deep forest features | MAE, RMSE, R2 |

This helps reviewers see that the decision layer is not only an LLM. It also
includes a conventional machine-learning baseline tied to the closest prior
cotton 3D paper.

## Source

- 3D reconstruction and characterization of cotton bolls in situ based on UAV
  technology: https://www.sciencedirect.com/science/article/pii/S0924271624000364
