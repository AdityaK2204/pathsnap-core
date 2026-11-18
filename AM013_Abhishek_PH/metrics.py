"""Candidate-level and observation-level evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def top1_per_fix(df, prob):
    """Return a frame with, per (trip, fix), the arc with highest score and whether it is correct."""
    d = df[["trip_id", "fix", "arc_id", "label"]].copy()
    d["prob"] = prob
    idx = d.groupby(["trip_id", "fix"], sort=False).prob.idxmax()
    best = d.loc[idx, ["trip_id", "fix", "arc_id", "label"]].rename(columns={"arc_id": "pred_arc", "label": "correct"})
    return best.reset_index(drop=True)


def evaluate(df, prob, threshold=0.5):
    """Metrics on candidate rows of fixes that have a reliable label."""
    m = df.label_valid.to_numpy()
    y = df.label.to_numpy()[m]
    p = np.clip(np.asarray(prob)[m], 1e-7, 1 - 1e-7)
    yhat = (p >= threshold).astype(int)
    best = top1_per_fix(df[m], p)
    hit = df[m].groupby(["trip_id", "fix"]).label.max()
    return dict(
        accuracy=accuracy_score(y, yhat),
        precision_macro=precision_score(y, yhat, average="macro", zero_division=0),
        recall_macro=recall_score(y, yhat, average="macro", zero_division=0),
        f1_macro=f1_score(y, yhat, average="macro"),
        f1_positive=f1_score(y, yhat),
        log_loss=log_loss(y, p, labels=[0, 1]),
        top1_accuracy=float(best.correct.mean()),
        candidate_miss_rate=float(1 - hit.mean()),
        n_fixes=int(len(best)),
        n_rows=int(m.sum()),
    )


def confusion(df, prob, threshold=0.5):
    m = df.label_valid.to_numpy()
    return confusion_matrix(df.label.to_numpy()[m], (np.asarray(prob)[m] >= threshold).astype(int), labels=[0, 1])
