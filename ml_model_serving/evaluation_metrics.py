"""Metrics for the melanoma classifier, computed the way the issue asks for.

Accuracy pools both classes, so on an imbalanced lesion set a model can post a
high number while missing most of the melanomas. Everything here is written in
terms of the melanoma class being **positive**, so `sensitivity` answers the
question that matters: of the melanomas in the set, how many did it catch.

Scores passed in are always P(melanoma). The model's raw output is
P(NotMelanoma) - see `melanoma_probability` in `model_prediction_service` - and
converting it is the caller's job, done once, in one place.

Nothing here loads a model or touches an image: it takes labels and scores and
returns numbers, so the maths can be tested on arrays written by hand. (The
package `__init__` still pulls in TensorFlow, which is what the stubs in
`tests/conftest.py` are for.)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# The minimums MODEL_CARD.md proposes. Targets, not results.
TARGET_SENSITIVITY = 0.85
TARGET_SPECIFICITY = 0.80
TARGET_AUC = 0.90


@dataclass(frozen=True)
class Counts:
    """A confusion matrix at one threshold. Melanoma is the positive class."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.tn + self.fp

    @property
    def sensitivity(self) -> float:
        """Of the melanomas, the share found. The number a miss is measured by."""
        return self.tp / self.positives if self.positives else float("nan")

    @property
    def specificity(self) -> float:
        """Of the non-melanomas, the share correctly left alone."""
        return self.tn / self.negatives if self.negatives else float("nan")

    @property
    def accuracy(self) -> float:
        total = self.positives + self.negatives
        return (self.tp + self.tn) / total if total else float("nan")


def confusion_at(y_true, y_score, threshold: float) -> Counts:
    """Confusion matrix, predicting melanoma when the score reaches `threshold`."""
    y_true = np.asarray(y_true).astype(bool)
    predicted = np.asarray(y_score, dtype=float) >= threshold
    return Counts(
        tp=int(np.count_nonzero(predicted & y_true)),
        fp=int(np.count_nonzero(predicted & ~y_true)),
        tn=int(np.count_nonzero(~predicted & ~y_true)),
        fn=int(np.count_nonzero(~predicted & y_true)),
    )


def threshold_for_sensitivity(y_true, y_score, target: float = TARGET_SENSITIVITY) -> float | None:
    """The highest threshold that still catches `target` of the melanomas.

    Highest, because every step down costs specificity. Returns None when no
    threshold reaches the target, which is itself the answer.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=float)
    if not y_true.any():
        return None
    candidates = sorted(set(y_score[y_true].tolist()), reverse=True)
    for threshold in candidates:
        if confusion_at(y_true, y_score, threshold).sensitivity >= target:
            return float(threshold)
    return None


def threshold_sweep(y_true, y_score, thresholds=None) -> list[dict]:
    """Sensitivity and specificity across thresholds, so the trade is visible."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
    rows = []
    for threshold in thresholds:
        counts = confusion_at(y_true, y_score, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": counts.sensitivity,
                "specificity": counts.specificity,
                "accuracy": counts.accuracy,
                "missed_melanomas": counts.fn,
            }
        )
    return rows


def evaluate(y_true, y_score, threshold: float = 0.5) -> dict:
    """Everything the model card is missing, for one set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    counts = confusion_at(y_true, y_score, threshold)

    both_classes_present = 0 < int(y_true.sum()) < len(y_true)
    auc = float(roc_auc_score(y_true, y_score)) if both_classes_present else float("nan")

    return {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
        "threshold": float(threshold),
        "sensitivity": counts.sensitivity,
        "specificity": counts.specificity,
        "accuracy": counts.accuracy,
        "auc_roc": auc,
        "counts": asdict(counts),
        "meets_targets": {
            "sensitivity": counts.sensitivity >= TARGET_SENSITIVITY,
            "specificity": counts.specificity >= TARGET_SPECIFICITY,
            "auc_roc": bool(auc >= TARGET_AUC) if both_classes_present else False,
        },
    }


def roc_points(y_true, y_score) -> list[dict]:
    """The ROC curve as data, for plotting or for the model card."""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float))
    return [
        {"threshold": float(t), "false_positive_rate": float(f), "sensitivity": float(s)}
        for f, s, t in zip(fpr, tpr, thresholds)
    ]


def _pct(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.4f}"


def markdown_report(result: dict, sweep: list[dict], model_name: str = "Xception") -> str:
    """The block to paste into MODEL_CARD.md, targets marked met or not."""
    counts = result["counts"]
    met = result["meets_targets"]
    tick = {True: "met", False: "**not met**"}

    lines = [
        f"### Measured on the held-out test split ({result['n']} images, "
        f"{result['prevalence']:.1%} melanoma)",
        "",
        f"Model: **{model_name}**. Melanoma is the positive class. "
        f"Operating threshold: {result['threshold']:.2f} on P(melanoma).",
        "",
        "| Metric | Value | Proposed minimum | |",
        "|---|---|---|---|",
        f"| Sensitivity (recall) | {_pct(result['sensitivity'])} | {TARGET_SENSITIVITY} | {tick[met['sensitivity']]} |",
        f"| Specificity | {_pct(result['specificity'])} | {TARGET_SPECIFICITY} | {tick[met['specificity']]} |",
        f"| AUC-ROC | {_pct(result['auc_roc'])} | {TARGET_AUC} | {tick[met['auc_roc']]} |",
        f"| Accuracy | {_pct(result['accuracy'])} | — | |",
        "",
        "Confusion matrix at that threshold:",
        "",
        "| | predicted melanoma | predicted not melanoma |",
        "|---|---|---|",
        f"| **is melanoma** | {counts['tp']} | {counts['fn']} (missed) |",
        f"| **is not melanoma** | {counts['fp']} | {counts['tn']} |",
        "",
        f"**{counts['fn']} melanomas were missed** at this threshold. That is the number "
        "accuracy hides.",
        "",
        "How sensitivity trades against specificity:",
        "",
        "| Threshold | Sensitivity | Specificity | Missed melanomas |",
        "|---|---|---|---|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['threshold']:.2f} | {_pct(row['sensitivity'])} | "
            f"{_pct(row['specificity'])} | {row['missed_melanomas']} |"
        )
    return "\n".join(lines)
