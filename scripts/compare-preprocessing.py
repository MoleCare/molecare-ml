"""Score the same images two ways: as /predict does, and as training did.

/predict runs tf.keras.applications.xception.preprocess_input, which maps pixels
to [-1, 1]. training/utils/data_loader.py trains and evaluates with
ImageDataGenerator(rescale=1./255), which is [0, 1]. Only one of those can be
what the weights expect.
"""
import os
import pathlib
import re
import sys

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("MODEL_PATH", "./cnn-models/xception/1")

from ml_model_serving.model_prediction_service import ModelPredictionService  # noqa: E402

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 500
ROOT = pathlib.Path("data/isic-eval")
ID = re.compile(r"ISIC_(\d+)")


def sample(folder, n):
    files = sorted(folder.glob("*.jpg"), key=lambda p: int(ID.search(p.name).group(1)))
    step = max(1, len(files) // n)
    return files[::step][:n]


paths, labels = [], []
for cls, label in (("Melanoma", 1), ("NotMelanoma", 0)):
    for p in sample(ROOT / cls, LIMIT // 2):
        paths.append(p)
        labels.append(label)

service = ModelPredictionService()


def score(path, mode):
    img = Image.open(path).convert("RGB").resize((299, 299))
    arr = np.asarray(img, dtype=np.float32)
    if mode == "serving":          # what /predict does
        arr = arr / 127.5 - 1.0
    else:                          # what training did
        arr = arr / 255.0
    batch = np.expand_dims(arr, 0).astype("float16")
    return 1.0 - float(service.predict_model(batch)[0][0])


results = {}
for mode in ("serving", "training"):
    scores = [score(p, mode) for p in paths]
    arr = np.array(scores)
    results[mode] = {
        "auc": roc_auc_score(labels, scores),
        "mean": float(arr.mean()),
        "sd": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

print(f"{len(paths)} images, {sum(labels)} melanoma\n")
print(f"{'preprocessing':<12} {'AUC':>7} {'mean':>8} {'sd':>7} {'min':>7} {'max':>7}")
for mode, r in results.items():
    print(f"{mode:<12} {r['auc']:>7.4f} {r['mean']:>8.4f} {r['sd']:>7.4f} {r['min']:>7.4f} {r['max']:>7.4f}")
