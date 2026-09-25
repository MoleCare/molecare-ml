#!/usr/bin/env python3
"""Measure sensitivity, specificity and AUC-ROC on the held-out test split.

Answers the question MODEL_CARD.md leaves open: accuracy is 0.9422, but how
often does this model miss a melanoma? See issue #23.

It predicts through the **serving path** - the same `ImageProcessor` and the
same `melanoma_probability` helper the API uses - so what is measured is what
is deployed, not a reimplementation of it that might get the polarity wrong.

Usage:

    scripts/fetch-model.sh                       # if you have no weights yet
    python scripts/evaluate.py \\
        --model-path ./cnn-models/xception/1 \\
        --test-dir /path/to/test \\
        --markdown doc/measured-metrics.md

`--test-dir` holds one folder per class. Folder names containing "mela" count
as melanoma, anything else as not melanoma; `--melanoma-dir` overrides that.
No test data lives in this repository, and none should.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# `python scripts/evaluate.py` puts scripts/ on sys.path, not the repository
# root, so `ml_model_serving` would not import however you invoked it. Add the
# root explicitly rather than making callers set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def find_images(test_dir: Path, melanoma_dir: str | None) -> list[tuple[Path, int]]:
    """Every image under `test_dir`, paired with 1 for melanoma and 0 for not."""
    labelled: list[tuple[Path, int]] = []
    for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        if melanoma_dir is not None:
            is_melanoma = class_dir.name == melanoma_dir
        else:
            is_melanoma = "mela" in class_dir.name.lower() and "not" not in class_dir.name.lower()
        for image in sorted(class_dir.rglob("*")):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                labelled.append((image, int(is_melanoma)))
    return labelled


def score_images(images: list[tuple[Path, int]], quiet: bool) -> tuple[list[int], list[float]]:
    """Run every image through the serving path. Returns (labels, P(melanoma))."""
    from ml_model_serving.image_processor import ImageProcessor
    from ml_model_serving.model_prediction_service import (
        ModelPredictionService,
        melanoma_probability,
    )

    processor = ImageProcessor()
    # The service owns loading. It knows the bundled artefact is a Keras 2
    # SavedModel and reaches for tf_keras, which Keras 3 cannot do - and using
    # it here is what makes "measured through the serving path" true of the
    # prediction as well as the preprocessing.
    service = ModelPredictionService()
    labels: list[int] = []
    scores: list[float] = []

    for index, (path, label) in enumerate(images, start=1):
        prepared = processor.prepare_input_from_bytes(path.read_bytes())
        raw = service.predict_model(prepared)[0][0]
        labels.append(label)
        scores.append(melanoma_probability(raw))
        if not quiet and index % 50 == 0:
            print(f"  {index}/{len(images)}", file=sys.stderr)

    return labels, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-path",
        help="SavedModel directory. Defaults to MODEL_PATH, the same variable the service reads.",
    )
    parser.add_argument("--test-dir", required=True, type=Path, help="Test split, one folder per class")
    parser.add_argument("--melanoma-dir", help="Name of the melanoma folder, if it is not obvious")
    parser.add_argument("--threshold", type=float, default=0.5, help="Operating threshold on P(melanoma)")
    parser.add_argument("--markdown", type=Path, help="Write the model card block here")
    parser.add_argument("--json", dest="json_out", type=Path, help="Write the raw numbers here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # The service reads MODEL_PATH when its class is defined, so this has to be
    # set before `ml_model_serving` is imported anywhere below.
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path

    from ml_model_serving import evaluation_metrics as metrics

    images = find_images(args.test_dir, args.melanoma_dir)
    if not images:
        print(f"No images under {args.test_dir}", file=sys.stderr)
        return 1
    positives = sum(label for _, label in images)
    if positives in (0, len(images)):
        print("The test split needs both classes to compute AUC.", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"{len(images)} images, {positives} melanoma", file=sys.stderr)

    labels, scores = score_images(images, args.quiet)

    result = metrics.evaluate(labels, scores, args.threshold)
    sweep = metrics.threshold_sweep(labels, scores)
    result["threshold_for_target_sensitivity"] = metrics.threshold_for_sensitivity(labels, scores)
    report = metrics.markdown_report(result, sweep)

    print(report)

    if args.markdown:
        args.markdown.write_text(report + "\n")
        print(f"\nWrote {args.markdown}", file=sys.stderr)
    if args.json_out:
        args.json_out.write_text(json.dumps({"summary": result, "sweep": sweep,
                                             "roc": metrics.roc_points(labels, scores)}, indent=2) + "\n")
        print(f"Wrote {args.json_out}", file=sys.stderr)

    # A red exit is the honest outcome when the model misses the target it set
    # itself. CI does not run this - there is no test data here - but a person
    # running it should not have to read the table to see that.
    return 0 if result["meets_targets"]["sensitivity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
