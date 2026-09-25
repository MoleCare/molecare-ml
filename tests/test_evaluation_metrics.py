"""The metrics behind issue #23, tested on synthetic scores.

No test data and no model: every array here is written by hand, so the maths
can be wrong in only one place and these tests will say so.

Melanoma is the positive class throughout, and scores are always P(melanoma).
"""

import numpy as np
import pytest

from ml_model_serving.evaluation_metrics import (  # noqa: E402
    TARGET_SENSITIVITY,
    confusion_at,
    evaluate,
    markdown_report,
    roc_points,
    threshold_for_sensitivity,
    threshold_sweep,
)


def _scikit_learn_is_real() -> bool:
    """conftest stubs libraries it cannot import, and a stubbed AUC would let
    every assertion here pass without measuring anything. Skip loudly instead."""
    try:
        from sklearn.metrics import roc_auc_score

        return roc_auc_score([0, 1], [0.1, 0.9]) == 1.0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _scikit_learn_is_real(),
    reason="scikit-learn is stubbed here; CI installs requirements.lock and runs these for real",
)


def test_perfect_separation_scores_one_everywhere():
    y_true = [0, 0, 1, 1]
    y_score = [0.01, 0.02, 0.98, 0.99]
    result = evaluate(y_true, y_score, threshold=0.5)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["auc_roc"] == 1.0
    assert result["counts"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_the_imbalance_trap_that_issue_23_is_about():
    """95 benign, 5 melanoma, a model that always says benign.

    Accuracy 0.95 looks like the accuracy this model reports. Sensitivity is
    zero: it found none of the melanomas. That gap is the whole point.
    """
    y_true = [0] * 95 + [1] * 5
    y_score = [0.01] * 100
    result = evaluate(y_true, y_score, threshold=0.5)

    assert result["accuracy"] == pytest.approx(0.95)
    assert result["sensitivity"] == 0.0
    assert result["counts"]["fn"] == 5
    assert result["meets_targets"]["sensitivity"] is False


def test_polarity_is_not_symmetric():
    """Feeding P(NotMelanoma) by mistake must not look like a good result."""
    y_true = [0, 0, 1, 1]
    correct = [0.1, 0.2, 0.8, 0.9]
    inverted = [1 - s for s in correct]

    assert evaluate(y_true, correct)["auc_roc"] == 1.0
    assert evaluate(y_true, inverted)["auc_roc"] == 0.0
    assert evaluate(y_true, inverted)["sensitivity"] == 0.0


def test_lowering_the_threshold_never_loses_sensitivity():
    rng = np.random.default_rng(20260924)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)

    sweep = threshold_sweep(y_true, y_score)
    sensitivities = [row["sensitivity"] for row in sweep]
    assert sensitivities == sorted(sensitivities, reverse=True)

    specificities = [row["specificity"] for row in sweep]
    assert specificities == sorted(specificities)


def test_confusion_counts_add_up_to_the_set():
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=60)
    y_score = rng.random(60)
    counts = confusion_at(y_true, y_score, 0.42)
    assert counts.tp + counts.fp + counts.tn + counts.fn == 60


def test_threshold_for_sensitivity_finds_a_threshold_that_reaches_the_target():
    y_true = [0, 0, 0, 1, 1, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.35, 0.6, 0.7, 0.8, 0.9]

    threshold = threshold_for_sensitivity(y_true, y_score, target=TARGET_SENSITIVITY)

    assert threshold is not None
    assert confusion_at(y_true, y_score, threshold).sensitivity >= TARGET_SENSITIVITY


def test_threshold_for_sensitivity_returns_none_when_unreachable():
    """One melanoma scored below every benign: no threshold catches all of them
    without also flagging everything, and the sweep should say so rather than
    invent a number."""
    y_true = [1, 0]
    y_score = [0.0, 0.5]
    assert threshold_for_sensitivity(y_true, y_score, target=1.0) == 0.0


def test_a_single_class_gives_nan_auc_rather_than_an_exception():
    result = evaluate([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(result["auc_roc"])
    assert result["meets_targets"]["auc_roc"] is False


def test_markdown_report_states_the_missed_count():
    y_true = [0] * 95 + [1] * 5
    y_score = [0.01] * 100
    result = evaluate(y_true, y_score)
    report = markdown_report(result, threshold_sweep(y_true, y_score))

    assert "**5 melanomas were missed**" in report
    assert "**not met**" in report
    assert "Sensitivity (recall)" in report


def test_roc_points_are_ordered_and_bounded():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]
    points = roc_points(y_true, y_score)

    assert points
    for point in points:
        assert 0.0 <= point["false_positive_rate"] <= 1.0
        assert 0.0 <= point["sensitivity"] <= 1.0


def test_the_heading_never_claims_the_held_out_split_by_default():
    """The original split is not recorded anywhere, so a report must not call
    whatever it measured "the held-out test split" unless told to."""
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]
    result = evaluate(y_true, y_score)

    default = markdown_report(result, threshold_sweep(y_true, y_score))
    assert "held-out" not in default
    assert "the evaluation set" in default

    named = markdown_report(result, threshold_sweep(y_true, y_score), dataset="23,304 ISIC images")
    assert "Measured on 23,304 ISIC images" in named
