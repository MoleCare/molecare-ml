# Measured metrics for the deployed Xception model

Measured on 25 September 2026 on yserver, with the weights published as
`weights-v1` and `scripts/evaluate.py`. This answers
[#23](https://github.com/MoleCare/molecare-ml/issues/23): sensitivity, specificity and
AUC-ROC, not only accuracy.

**Read "What these numbers cannot tell you" before quoting any of them.** The most
important caveat is that this is not the split the published 0.9422 came from, because
that split is not recorded anywhere.

## The short version

- **The model ranks images reasonably well and is thresholded badly.** At the default
  threshold of 0.50 it finds about one melanoma in three.
- **On recent melanomas it is below target at every threshold.** On images it is
  unlikely to have seen, AUC is 0.8279 against a target of 0.90, and no operating
  point meets both the sensitivity and the specificity target.
- **It does much better on old images than new ones.** AUC falls from 0.9371 on the
  oldest third of the archive to 0.8279 on the newest. That gap is consistent with the
  model having trained on some of the old images, or with newer images being harder.
  The data cannot separate the two.
- **The risk level people see does not come from this model at all.** In production
  `/analyze` never loads the model, so its risk bands run on the ABCDE image heuristic
  alone — which, on 2,000 of these images, separates melanoma from benign moles barely
  better than chance (AUC 0.5378) and can never reach `urgent`. See
  [#102](https://github.com/MoleCare/molecare-ml/issues/102) and
  [the escalation bands](#the-escalation-bands) below.

## What was measured

| | |
|---|---|
| Model | Xception, `weights-v1`, loaded through `ModelPredictionService` exactly as the API loads it |
| Positives | 11,304 ISIC images with `diagnosis_2` "Malignant melanocytic proliferations (Melanoma)" — every one the archive returned |
| Negatives | 12,000 ISIC images with `diagnosis_1:Benign AND melanocytic:true` — benign moles, the hard negative |
| Preprocessing | the serving path: `ImageProcessor` and `melanoma_probability` |
| Hardware | CPU only, 32 cores |

Negatives are benign **melanocytic** lesions on purpose. As [data.md](data.md) says,
benign non-melanocytic lesions are an easier negative and would flatter the score.

## Results

AUC-ROC does not depend on the threshold. Sensitivity and specificity are at 0.50.

| Set | Images | AUC-ROC | Sensitivity | Specificity |
|---|---|---|---|---|
| All | 23,304 | 0.9113 | 0.3184 | 0.9828 |
| Oldest third by `isic_id` | 7,756 | **0.9371** | 0.3229 | 0.9860 |
| Newest third by `isic_id` | 7,756 | **0.8279** | 0.3506 | 0.9667 |
| *Proposed minimum* | | *0.90* | *0.85* | *0.80* |

Confusion matrix on all 23,304 at 0.50:

| | predicted melanoma | predicted not melanoma |
|---|---|---|
| **is melanoma** | 3,599 | **7,705 missed** |
| **is not melanoma** | 207 | 11,793 |

### How the threshold changes things

| Threshold | All: sens / spec | Oldest: sens / spec | Newest: sens / spec |
|---|---|---|---|
| 0.05 | 0.9448 / 0.6984 | 0.9483 / 0.7770 | 0.9308 / 0.4710 |
| **0.10** | **0.8610 / 0.8116** | **0.8653 / 0.8758** | 0.8349 / 0.6250 |
| 0.15 | 0.7720 / 0.8689 | 0.7678 / 0.9163 | 0.7417 / 0.7278 |
| 0.20 | 0.6972 / 0.9058 | 0.6938 / 0.9395 | 0.6701 / 0.8010 |
| 0.50 | 0.3184 / 0.9828 | 0.3229 / 0.9860 | 0.3506 / 0.9667 |
| 0.60 | 0.2374 / 0.9907 | 0.2463 / 0.9908 | 0.2865 / 0.9830 |
| 0.80 | 0.1211 / 0.9973 | 0.1203 / 0.9975 | 0.1837 / 0.9952 |

At 0.10 both targets are met on the full set and on the oldest third. **They are not met
on the newest third**: sensitivity 0.8349, specificity 0.6250. So moving the threshold
recovers a great deal on familiar images, and much less on recent ones. Choosing a
threshold from the full-set table alone would overstate what the model does on new
photographs.

## The escalation bands

`ml_model_serving/mole_analysis_service.py`:

```python
if combined_score >= 0.65 or ml_risk >= 0.8:
    risk_level = "very_high"; urgent = True
elif combined_score >= 0.45 or ml_risk >= 0.6:
    risk_level = "high"
```

**In production the model never reaches these bands.** `prediction_controller.py` and
`evolution_analysis_service.py` build `MoleAnalysisService()` with no model path, so
`ml_risk` stays at its placeholder of 0.5 and the model contributes nothing. (Passing the
path would not help on its own: `_load_ml_model` uses `tf.keras.models.load_model`, which
Keras 3 refuses for this artefact.)

With the model absent, the combined score is half the ABCDE score. The ABCDE score is
capped at 1.0, so the combined score can never exceed 0.5 — and `very_high`, the only
band that sets `urgent`, needs 0.65. **`urgent` cannot fire.**

Measured by running `analyze_mole` exactly as production builds it, on 1,000 melanomas
and 1,000 benign moles from this set:

| | Melanomas | Benign moles |
|---|---|---|
| `low` | 251 | 336 |
| `moderate` | 749 | 663 |
| `high` | **0** | 1 |
| `very_high` / `urgent` | 0 | 0 |
| `melanoma_probability` returned | 0.5, every time | 0.5, every time |

AUC of the score behind the bands: **0.5378**. The label people see tells a melanoma from
a benign mole barely better than chance, and calls 66% of benign moles "moderate".

**If the model were connected**, the model-only figures from the tables above would
apply: `ml_risk >= 0.8` catches 12.1% of melanomas across all images and 18.4% on recent
ones; `ml_risk >= 0.6` catches 23.7% and 28.7%. So connecting it would switch on an
`urgent` banner that stays silent for most melanomas.

What `/analyze` should promise is a product and clinical decision, tracked in
[#100](https://github.com/MoleCare/molecare-ml/issues/100) and
[#102](https://github.com/MoleCare/molecare-ml/issues/102). Reproduce with
`python scripts/measure-analyze-bands.py 1000`.

## What these numbers cannot tell you

**This is not the original held-out split.** `DataLoader` reads three folders that
already existed on disk, and nothing recorded which images were in them. The 0.9422
accuracy and these figures come from different images and cannot be compared directly.

**Some of these images may have been in training.** The model card says the training
data was ISIC; any ISIC image could have been seen, and without a manifest there is no
way to exclude them. That would inflate these numbers. The newest third is the best
available guard: those melanomas are the least likely to have been downloaded when the
model was trained.

**The newest third is only half new.** `isic image download --limit` returns images in
ascending `isic_id` order, so all 12,000 negatives are older images, `ISIC_0000000` to
`ISIC_0015207`. In the newest third the melanomas are recent and the benign moles are
not. If the model remembers those moles, they are easier negatives than truly new ones,
so **0.8279 is an optimistic figure** for fully unseen images. It is below 0.90 even so.

**Prevalence is constructed.** The set is 48.5% melanoma because it was built that way.
Sensitivity, specificity and AUC do not depend on prevalence; accuracy does, which is
why accuracy is not reported as a headline here.

**Dermoscopy only.** These are dermoscopic images. People send MoleCare phone photos,
and performance on those is still not characterised ([#27](https://github.com/MoleCare/molecare-ml/issues/27)).

**Skin tone is still unmeasured.** ISIC does not reliably carry Fitzpatrick type, so
nothing here addresses [#10](https://github.com/MoleCare/molecare-ml/issues/10).

## A mismatch that turned out not to matter

`/predict` normalises with `tf.keras.applications.xception.preprocess_input`, which maps
pixels to [-1, 1]. Training used `ImageDataGenerator(rescale=1./255)`, which is [0, 1].
They should agree. Scoring the same 800 images both ways gave AUC 0.9124 and 0.9097, so
the difference does not explain anything above. It is still worth making the two paths
agree, because the same photo gets a slightly different number depending on the
endpoint.

## Reproducing this

```bash
scripts/fetch-model.sh
pip install isic-cli

isic image download --limit 0 \
  --search 'diagnosis_2:"Malignant melanocytic proliferations (Melanoma)"' data/isic-raw/mel-all
isic image download --limit 12000 \
  --search 'diagnosis_1:Benign AND melanocytic:true' data/isic-raw/ben-all

mkdir -p data/isic-eval
ln -s ../isic-raw/mel-all data/isic-eval/Melanoma
ln -s ../isic-raw/ben-all data/isic-eval/NotMelanoma

python scripts/evaluate.py --test-dir data/isic-eval --dataset-name "23,304 ISIC images"
python scripts/make-era-sets.py        # oldest and newest thirds by isic_id
python scripts/compare-preprocessing.py 800
python scripts/measure-analyze-bands.py 1000   # /analyze exactly as production builds it
```

The archive changes over time, so a later run will not return exactly these images.

## What would make these numbers trustworthy

1. **A dataset manifest** ([#33](https://github.com/MoleCare/molecare-ml/issues/33)) —
   the list of image ids used in training, so a test set can exclude them by
   construction instead of by guesswork.
2. **Recent negatives as well as recent positives** — download all 48,974 benign moles
   and take the newest, so the newest third is new on both sides.
3. **Phone photos and skin-tone labels** — the two gaps this run cannot touch.
