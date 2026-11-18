# Model training and comparison: Abhishek P H (PES1UG24AM013)

Seven candidate classifiers from different families, trained on the 80 training trips
with 5-fold GroupKFold (trip = group) and compared on the 10 student-testing trips.

| File | Purpose |
|---|---|
| `model_zoo.py` | model definitions and initial hyperparameters |
| `training.py` | fitting, incl. epoch / boosting-round loss histories (MLP, LightGBM, HistGB) |
| `metrics.py` | Macro F1, precision, recall, log loss, observation-level Top-1, candidate miss rate |
| `train_compare.py` | CV + student-test evaluation of every model, model selection |
| `plots.py` | comparison charts, confusion matrices, training histories |
| `extra_analysis.py` | training vs CV vs test scores (over/under-fitting), per-trip Top-1, per-class metrics, feature importance |

Run (from the repository root, after `build_dataset`):

```
python -m AM013_Abhishek_PH.train_compare
python -m AM013_Abhishek_PH.plots
```

Outputs: `results/models/` (`model_comparison.csv`, `cv_fold_scores.csv`, `experiment_log.jsonl`,
`selected_model.json`, `plots/`), saved models in `artifacts/models/`.
