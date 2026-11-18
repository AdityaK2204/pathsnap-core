# PathSnap core: data processing, features and HMM/Viterbi route reconstruction

Part of **PathSnap** (UE24CS352A Machine Learning, PES University RR campus, team PT-004).
This repository holds the part of the project built by **Aditya Kalloli (PES1UG24AM015)**:
everything from the raw trip files to a decoded, connected route.

```
raw .track/.nodes/.arcs -> cleaning + per-trip metric projection -> road graph + spatial index
 -> candidate arcs (<= 12 within 60 m) -> 21 features per (fix, arc) -> [classifier]
 -> HMM / Viterbi over network driving distances -> per-fix arcs + reconstructed route
```

The candidate classifier itself (model comparison, Bayesian tuning), the API/UI and the
EDA/maps were built by the other team members in the main team repository; this package
is what all of them import, so preprocessing is identical for training, evaluation and the API.

## Contents

| File | Purpose |
|---|---|
| `AM015_Aditya_Kalloli/config.py` | paths (`PATHSNAP_DATA` env var) and pipeline constants |
| `data_io.py` | reading `.track/.nodes/.arcs/.route`; `load_trip` never reads the route |
| `geo.py` | per-trip local metric projection, bearings, point-to-segment distance |
| `road_network.py` | directed graph per trip, KD-tree over densified arcs, bounded Dijkstra |
| `label_alignment.py` | monotone dynamic programme that assigns one reference arc to every GPS fix |
| `candidates.py` | top-K arcs within a radius for each fix (fallback to the nearest arcs) |
| `features.py` | 21 features per (fix, candidate) pair: distance, heading, temporal window, topology, kinematics |
| `preprocess.py` | the single `prepare_trip` used everywhere |
| `build_dataset.py` | builds the candidate/feature tables and `results/preprocessing/label_quality.csv` |
| `hmm_viterbi.py` | emissions, transitions, Viterbi with breaks, shortest-path route reconstruction |
| `tune_decoder.py` | grid search of beta / emission weight on training trips (out-of-fold probabilities) |
| `evaluate_hmm.py` | classifier-only vs classifier + HMM on the student-testing trips |
| `error_analysis.py` | splits remaining errors into reverse-twin / adjacent-arc / other |
| `hmm_plots.py` | decoder grid and per-trip gain plots |
| `pipeline.py` | `MapMatcher` (used by the API) and server-side scoring |
| `route_metrics.py` | route continuity, reachable continuity, route precision / recall / F1 |

`results/` and `artifacts/decoder_config.json` are the outputs of these scripts.

## Method in short

- **Projection**: trips are spread over the whole globe, so each trip gets a local equirectangular
  projection centred on its median fix; all distances are in metres.
- **Labels**: the `.route` file is an ordered arc list, not per-fix labels. A monotone DP aligns fixes to
  route positions (minimum total distance, never moving backwards); fixes > 50 m from their aligned arc
  are left unlabelled (97.5 % of training fixes keep a label).
- **Candidates**: arcs within 60 m, at most 12; candidate miss rate 0.10 % on student-test.
- **HMM**: states are the candidate arcs; emission = w * log(normalised classifier probability);
  transition = -|d_route - d_gc| / beta, with d_route from a bounded Dijkstra on the directed graph;
  the chain is split at gaps > 120 s / 3 km and restarted when nothing is reachable.
  Only beta * w changes the Viterbi path; the grid search picked beta = 1, w = 0.5.

## Results (10 student-testing trips)

| | Top-1 accuracy | Route continuity | Route F1 (length-weighted) |
|---|---|---|---|
| classifier only (tuned LightGBM) | 0.9117 | 0.9554 | - |
| classifier + HMM/Viterbi | **0.9530** | **0.9780** | **0.9892** |

HMM decoding improved all 10 trips. Remaining errors: 2.0 % reverse direction of the correct road
(mostly at low speed), 2.1 % neighbouring arc at an arc boundary, 0.6 % other roads
(`results/hmm/error_categories_summary.csv`).

## Running

Python 3.10+.

```
pip install -r requirements.txt
set PATHSNAP_DATA=C:\path\to\folder\with\train_and_student_test
python -m AM015_Aditya_Kalloli.build_dataset
python -m AM015_Aditya_Kalloli.evaluate_hmm      # needs artifacts/final_model.joblib from the team repo
python -m AM015_Aditya_Kalloli.error_analysis
python -m AM015_Aditya_Kalloli.hmm_plots
```

`tune_decoder.py` additionally needs the out-of-fold classifier probabilities produced by the
Bayesian-optimisation step of the team repository. The raw dataset is not included.
