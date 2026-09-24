# Model Card — MoleCare Melanoma Classifier

Following the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) framework.

> ## ⚠️ Not a medical device
> This model is **not** a diagnostic tool and must not be used for diagnosis, triage, or any
> decision about a person's care. It has not been clinically validated, has no regulatory
> approval, and has not been evaluated on the populations it would need to serve. Anyone
> concerned about a skin lesion should see a qualified clinician.

## Model details

| | |
|---|---|
| **Architecture** | Xception, ImageNet-pretrained, fine-tuned |
| **Task** | Binary image classification — Melanoma vs NotMelanoma |
| **Input** | RGB dermoscopic image, 299×299 |
| **Parameters** | ~20.9M |
| **Framework** | TensorFlow / Keras, served via Flask |
| **Licence** | Apache-2.0 (weights and code); training data licensed separately |
| **Maintainer** | MoleCare — info@molecare.co.uk |

## Training data

Dermoscopic images from the **[ISIC Archive](https://www.isic-archive.com/)**, with augmentation.
**No MoleCare user or patient images were used**, and none are present in this repository.

ISIC images carry a range of licences (CC-0, CC-BY, CC-BY-NC) depending on the contributing
collection. Verify the terms for your use case before redistributing derived artefacts —
a CC-BY-NC source restricts commercial use.

## Evaluation

Measured on a held-out test split, 50 epochs, batch size 16:

| Model | Test accuracy | Test loss | Params (M) | Input |
|---|---|---|---|---|
| **Xception** *(deployed)* | **0.9422** | 0.1777 | 20.9 | 299×299 |
| InceptionV3 | 0.9416 | 0.1848 | 21.9 | 299×299 |
| InceptionResNetV2 | 0.9384 | 0.2136 | 54.4 | 299×299 |
| DenseNet201 | 0.9369 | 0.1974 | 18.4 | 224×224 |
| VGG16 | 0.8270 | 0.4212 | 14.7 | 224×224 |

### What these numbers do not tell you

**Only accuracy was measured. Sensitivity, specificity and AUC-ROC were not.**

For a melanoma classifier this is the limitation that matters most. Accuracy on a
class-imbalanced dataset can look strong while the model misses a large share of actual
melanomas — and a false negative here is the dangerous error. Until sensitivity and AUC are
measured and published, **94% accuracy should not be read as "94% reliable"**.

The comparison notebook records the metrics that should gate any future deployment:

| Metric | Why it matters | Proposed minimum |
|---|---|---|
| Sensitivity (recall) | Missing a melanoma is the harmful failure | ≥ 0.85 |
| AUC-ROC | Best single discrimination measure | ≥ 0.90 |
| Specificity | Limits unnecessary biopsies and alarm | ≥ 0.80 |

These are **targets, not results.** Any figures in the notebooks labelled "expected" are
projections from proposed training changes, not measurements.

Turning them into results now needs only the held-out test split and the deployed
weights, because the measurement itself is written:

```bash
scripts/fetch-model.sh
python scripts/evaluate.py --model-path ./cnn-models/xception/1 --test-dir <test split>
```

`scripts/evaluate.py` predicts through the serving path — the same image
preprocessing and the same `melanoma_probability` helper the API uses — and prints
sensitivity, specificity, AUC-ROC, the confusion matrix, and how the two trade
across thresholds, in a block shaped to replace this section. It exits non-zero when
sensitivity misses the 0.85 target above. Nothing in this repository carries test
images, so the run has to happen somewhere that has them ([#23](https://github.com/MoleCare/molecare-ml/issues/23)).

## Known limitations and biases

- **Skin tone.** Public dermoscopic datasets over-represent lighter skin, and this model
  inherits that. Performance across Fitzpatrick types is **unmeasured** — see the open
  [bias evaluation issue](https://github.com/MoleCare/MoleCare-ML/issues). This is the most
  important open problem in the repository.
- **Image quality.** Trained on dermoscopic images. Consumer phone photos differ in lighting,
  focus and scale, and performance on them is not characterised.
- **Binary framing.** Melanoma vs not-melanoma collapses many diagnoses into one negative class.
  It cannot distinguish among non-melanoma conditions.
- **No calibration.** Output scores are not calibrated probabilities and should not be presented
  to a person as a percentage likelihood.

## Intended and out-of-scope use

**Intended:** research, education, and prototyping behind appropriate clinical disclaimers.

**Out of scope:** autonomous diagnosis; triage without a clinician; any regulatory or clinical
claim; use as evidence in a care decision.

## How to cite the data

Please credit the ISIC Archive when publishing work derived from these models.
