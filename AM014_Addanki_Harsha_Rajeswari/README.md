# Bayesian optimisation, API, UI and deployment: Addanki Harsha Rajeswari (PES1UG24AM014)

| File | Purpose |
|---|---|
| `bayes_opt.py` | Optuna TPE search on the selected model, tuned vs untuned comparison, final model export, out-of-fold probabilities for the decoder |
| `tuning_plots.py` | optimisation history, parameter importance, tuned vs untuned charts |
| `api/main.py` | FastAPI backend: `/health`, `/trips`, `/predict` |
| `api/schemas.py` | request validation |
| `ui/app.py` | Streamlit evaluator UI (laptop 2) |
| `ui/requirements.txt` | the only packages laptop 2 needs |
| `api_check_all_trips.py` | sends every served trip through `/predict` and prints a summary |

Run (from the repository root):

```
python -m AM014_Addanki_Harsha_Rajeswari.bayes_opt --trials 30
python -m AM014_Addanki_Harsha_Rajeswari.tuning_plots
```

Deployment is described in the main README.
