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

That table is the only thing measured when the model was trained, and the split it used
was never recorded. **94% accuracy should not be read as "94% reliable"**: accuracy on a
class-imbalanced set can look strong while the model misses most melanomas.

### Sensitivity, specificity and AUC — measured

Measured on 25 September 2026 against ISIC dermoscopic images, through the serving path.
Full method, sweeps and caveats: **[doc/measured-metrics.md](doc/measured-metrics.md)**.

| Set | Images | AUC-ROC | Sensitivity @ 0.50 | Specificity @ 0.50 |
|---|---|---|---|---|
| All | 23,304 | 0.9113 | 0.3184 | 0.9828 |
| Oldest third by `isic_id` | 7,756 | 0.9371 | 0.3229 | 0.9860 |
| **Newest third by `isic_id`** | 7,756 | **0.8279** | **0.3506** | 0.9667 |
| *Proposed minimum* | | *≥ 0.90* | *≥ 0.85* | *≥ 0.80* |

What they say:

- **At the default threshold of 0.50 the model finds about one melanoma in three.**
  On all 23,304 images it missed 7,705 of 11,304.
- **On recent melanomas it misses the targets at every threshold.** The newest third is
  the best available guard against images the model trained on, and its AUC of 0.8279 is
  below 0.90. It is also an optimistic figure, because the benign images in that third are
  older ones.
- **A lower threshold helps on familiar images far more than on new ones.** At 0.10 both
  targets are met on the full set (0.861 / 0.812) but not on the newest third
  (0.835 / 0.625).
- **Old images score much better than new ones** — AUC 0.9371 against 0.8279. That fits
  the model having trained on some of the old images, or newer images being harder. The
  data cannot separate the two, because nothing recorded what it trained on.

**This is not the original held-out split** and is not directly comparable to the 0.9422
above. It is dermoscopy only, and says nothing about phone photos or skin tone.

The comparison notebook's minimums still stand as the bar for deployment:

| Metric | Why it matters | Proposed minimum |
|---|---|---|
| Sensitivity (recall) | Missing a melanoma is the harmful failure | ≥ 0.85 |
| AUC-ROC | Best single discrimination measure | ≥ 0.90 |
| Specificity | Limits unnecessary biopsies and alarm | ≥ 0.80 |

The deployed model does not meet them on recent images.

## Known limitations and biases

- **Skin tone.** Public dermoscopic datasets over-represent lighter skin, and this model
  inherits that. Performance across Fitzpatrick types is **unmeasured** — see the open
  [bias evaluation issue](https://github.com/MoleCare/MoleCare-ML/issues). This is the most
  important open problem in the repository.
- **Image quality.** Trained on dermoscopic images. Consumer phone photos differ in lighting,
  focus and scale, and performance on them is not characterised.

Those two limitations share a candidate answer. Google's [SCIN dataset](https://github.com/google-research-datasets/scin)
is open access, is made of **consumer phone photos** rather than dermoscopy, and carries
estimated Fitzpatrick skin type and Monk Skin Tone labels — which is what measuring the
first two bullets needs. It will not close the sensitivity gap above: roughly 89% of it is
allergic, infectious and inflammatory conditions, and it does not cover melanoma. It can
show whether this model behaves differently across skin tones on the kind of photo people
actually send; it cannot show how many melanomas it misses.
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
