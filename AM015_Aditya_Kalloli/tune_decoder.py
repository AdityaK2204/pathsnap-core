"""Grid search of the HMM decoder parameters on the training trips.

Uses out-of-fold classifier probabilities (produced by bayes_opt.py) so the
decoder is tuned on probabilities that look like those it will see on unseen
trips. Student-testing trips are only reported, not used for the choice.

Usage (from the repository root):
    python -m AM015_Aditya_Kalloli.tune_decoder
"""
import itertools
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config
from .data_io import list_trips, load_route, load_trip
from .hmm_viterbi import DecoderParams, candidate_lists, transition_costs, viterbi
from .label_alignment import make_labels
from .preprocess import prepare_trip

BETAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
EMISSION_WEIGHTS = [0.25, 0.5, 1.0, 2.0]
OUT = config.RESULTS_DIR / "hmm"


def trip_grid(trip_id, prob_df):
    trip = load_trip(config.TRAIN_DIR, trip_id)
    p = prepare_trip(trip)
    pr = prob_df.loc[trip_id]
    probs = pr.set_index(["fix", "arc_id"]).prob.reindex(
        pd.MultiIndex.from_arrays([p.cands.fix, p.cands.arc_id])).to_numpy()
    lab = make_labels(p.xy, p.net.node_xy, trip.arcs, load_route(config.TRAIN_DIR, trip_id))
    v = lab["label_valid"]
    cl = candidate_lists(p.cands, probs, len(p.xy))
    costs, _ = transition_costs(cl, p.xy, p.t, p.net)
    clf = np.array([c[0][int(np.argmax(c[2]))] for c in cl])
    rows = [dict(trip_id=trip_id, beta=np.nan, emission_weight=np.nan, decoder="classifier_only",
                 correct=int((clf == lab["label_arc"])[v].sum()), n=int(v.sum()))]
    for beta, w in itertools.product(BETAS, EMISSION_WEIGHTS):
        ch, br = viterbi(cl, costs, DecoderParams(beta=beta, emission_weight=w))
        arcs = np.array([cl[i][0][ch[i]] for i in range(len(cl))])
        rows.append(dict(trip_id=trip_id, beta=beta, emission_weight=w, decoder="hmm",
                         correct=int((arcs == lab["label_arc"])[v].sum()), n=int(v.sum()), breaks=br))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prob_df = pd.read_parquet(config.CACHE_DIR / "oof_train_probabilities.parquet").set_index("trip_id")
    ids = list_trips(config.TRAIN_DIR)
    res = Parallel(n_jobs=8, verbose=5)(delayed(trip_grid)(t, prob_df.loc[[t]]) for t in ids)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT / "decoder_grid_per_trip.csv", index=False)

    hmm = df[df.decoder == "hmm"].groupby(["beta", "emission_weight"])[["correct", "n"]].sum()
    hmm["top1_pooled"] = hmm.correct / hmm.n
    per_trip = df[df.decoder == "hmm"].assign(acc=lambda d: d.correct / d.n.clip(lower=1)) \
        .groupby(["beta", "emission_weight"]).acc.mean()
    hmm["top1_mean_over_trips"] = per_trip
    hmm = hmm.reset_index()
    # the Viterbi argmax only depends on beta * emission_weight (scaling the whole score changes nothing)
    hmm["beta_x_weight"] = hmm.beta * hmm.emission_weight
    hmm = hmm.sort_values(["top1_pooled", "beta"], ascending=[False, True])
    hmm.to_csv(OUT / "decoder_grid_summary.csv", index=False)
    clf = df[df.decoder == "classifier_only"]
    clf_acc = clf.correct.sum() / clf.n.sum()

    best = hmm.iloc[0]
    cfg = dict(beta=float(best.beta), emission_weight=float(best.emission_weight), prob_floor=1e-4,
               selection="pooled observation-level Top-1 on training trips with out-of-fold probabilities",
               train_top1_hmm=float(best.top1_pooled), train_top1_classifier_only=float(clf_acc),
               gap_break_s=config.GAP_BREAK_S, gap_break_m=config.GAP_BREAK_M)
    (config.ARTIFACTS_DIR / "decoder_config.json").write_text(json.dumps(cfg, indent=2))
    print(hmm.head(10).round(4).to_string(index=False))
    print("classifier-only (OOF) top-1:", round(clf_acc, 4))
    print("selected:", cfg)


if __name__ == "__main__":
    main()
