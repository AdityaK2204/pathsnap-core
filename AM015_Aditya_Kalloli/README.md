# Core pipeline and HMM/Viterbi: Aditya Kalloli (PES1UG24AM015)

Raw trip files → cleaned GPS → candidate road segments → features → (classifier) → HMM/Viterbi route.
Every other part of the project imports this package, so preprocessing is identical
for training, evaluation and the API.

| File | Purpose |
|---|---|
| `config.py` | paths (`PATHSNAP_DATA` env var) and pipeline constants |
| `data_io.py` | reading `.track/.nodes/.arcs/.route`; `load_trip` never reads the route |
| `geo.py` | per-trip local metric projection, bearings, point-to-segment distance |
| `road_network.py` | per-trip directed graph, spatial index of densified arcs, bounded Dijkstra |
| `label_alignment.py` | monotone DP that assigns one reference arc to each GPS fix |
| `candidates.py` | top-K arcs within a radius for each fix (fallback to nearest arcs) |
| `features.py` | 21 features per (fix, candidate) pair |
| `preprocess.py` | single `prepare_trip` used everywhere |
| `build_dataset.py` | builds `data_cache/*.parquet` and `results/preprocessing/label_quality.csv` |
| `hmm_viterbi.py` | emissions, transitions, Viterbi with breaks, route reconstruction |
| `tune_decoder.py` | grid search of `beta` / emission weight on training trips (out-of-fold probabilities) |
| `evaluate_hmm.py` | classifier-only vs classifier + HMM on student-test trips |
| `error_analysis.py` | splits remaining errors into reverse-twin / adjacent-arc / other |
| `hmm_plots.py` | decoder grid, beta x weight curve, per-trip gain plots |
| `pipeline.py` | `MapMatcher` used by the API, plus server-side scoring |
| `route_metrics.py` | continuity, reachable continuity, route precision/recall/F1 |

Run (from the repository root):

```
python -m AM015_Aditya_Kalloli.build_dataset
python -m AM015_Aditya_Kalloli.tune_decoder      # needs artifacts from bayes_opt.py
python -m AM015_Aditya_Kalloli.evaluate_hmm
```
