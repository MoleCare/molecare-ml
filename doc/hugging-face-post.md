# Hugging Face post

Written for huggingface.co/posts, which caps a post at about 2,000 characters. It is a short form
of doc/94-percent-accurate.md; the table and every number come from there, so the two must be
corrected together.

---

**Our melanoma classifier is 94.2% accurate. That number is close to meaningless, and we put that in the model card.**

We published the weights, the training notebooks and the model card. The headline number is 94.2% test accuracy. Here is why you should not trust it.

**We measured the wrong thing.** Accuracy is the fraction of predictions correct, pooled across both classes. For melanoma detection what matters is **sensitivity** — of the melanomas, how many did we catch? — and specificity, and AUC-ROC. We measured none of them. A model that called everything "not melanoma" would still score well on a skewed test set, and accuracy would not tell you.

**The architecture choice was noise.** Five architectures, 50 epochs, batch size 16:

| Model | Test accuracy | Params |
|---|---|---|
| **Xception** *(deployed)* | **0.9422** | 20.9M |
| InceptionV3 | 0.9416 | 21.9M |
| InceptionResNetV2 | 0.9384 | 54.4M |
| DenseNet201 | 0.9369 | 18.4M |
| VGG16 | 0.8270 | 14.7M |

Xception beat InceptionV3 by six ten-thousandths. That gap is noise. We deployed it anyway — a defensible engineering call and an indefensible scientific one, and it's worth being clear which is which.

**The gap that matters most is unmeasured.** Performance across Fitzpatrick skin types. Dermoscopic archives under-represent darker skin, so a model trained on them may well be worse exactly where a missed melanoma is already more often fatal. We have not measured this. It is [an open issue](https://github.com/MoleCare/molecare-ml/issues/10) and the contribution I would most like.

Xception fine-tuned on [ISIC Archive](https://www.isic-archive.com/) dermoscopic images, 299×299 RGB. No user photos were used in training or are in the repo.

**This is not a medical device and it does not diagnose anything.** If a publishable number is the goal, publish what you didn't measure alongside it.

https://github.com/MoleCare/molecare-ml
