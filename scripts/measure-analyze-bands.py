"""What /analyze tells people, measured the way production runs it.

prediction_controller.py builds MoleAnalysisService() with no model path, so
this does the same. Each image goes through analyze_mole exactly as a request
would, and we record the risk level and the urgent flag it returns.
"""
import json
import pathlib
import random
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml_model_serving.mole_analysis_service import MoleAnalysisService  # noqa: E402

PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 20260925
ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "isic-eval"

service = MoleAnalysisService()  # exactly as production constructs it
print(f"ml model loaded: {service.ml_model is not None}", flush=True)

rng = random.Random(SEED)
rows = []
started = time.time()
for cls, label in (("Melanoma", 1), ("NotMelanoma", 0)):
    files = sorted((ROOT / cls).glob("*.jpg"))
    for path in rng.sample(files, min(PER_CLASS, len(files))):
        r = service.analyze_mole(path.read_bytes())
        rows.append({
            "label": label,
            "risk": r.combined_risk_level,
            "score": r.combined_risk_score,
            "urgent": bool(r.urgent_referral),
            "melanoma_probability": r.melanoma_probability,
            "abcde": r.abcde_score.total_score,
        })

print(f"{len(rows)} images in {time.time() - started:.0f}s\n")
for label, name in ((1, "melanoma"), (0, "benign")):
    sub = [r for r in rows if r["label"] == label]
    levels = Counter(r["risk"] for r in sub)
    print(f"{name:9} n={len(sub)}  " + "  ".join(f"{k}={levels.get(k, 0)}" for k in ("low", "moderate", "high", "very_high")))
    print(f"          urgent={sum(r['urgent'] for r in sub)}  "
          f"distinct melanoma_probability values={sorted({r['melanoma_probability'] for r in sub})}")
    print(f"          max combined score={max(r['score'] for r in sub):.3f}  max abcde={max(r['abcde'] for r in sub):.3f}")

mel = [r for r in rows if r["label"] == 1]
for band in ("moderate", "high"):
    order = ["low", "moderate", "high", "very_high"]
    at_or_above = [r for r in mel if order.index(r["risk"]) >= order.index(band)]
    print(f"\nmelanomas at '{band}' or above: {len(at_or_above)}/{len(mel)} = {len(at_or_above) / len(mel):.3f}")

pathlib.Path("data/analyze-bands.json").write_text(json.dumps(rows))
