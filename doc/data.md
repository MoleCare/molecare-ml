# Getting the data

Nothing in this repository ships images. The training code expects a directory
of them in a specific shape, and until now nothing said what that shape is,
which is why every open issue that needs the held-out test split has been
impossible to pick up from outside.

## What the code expects

`DataLoader` takes `data_dir` and reads three directories under it. Each holds
one subdirectory per class, because Keras derives the labels from the folder
names. `ModelConfig.data_dir` defaults to `data/`, so that is the conventional
location.

```
data/
├── training/
│   ├── Melanoma/
│   └── NotMelanoma/
├── validation/
│   ├── Melanoma/
│   └── NotMelanoma/
└── testing/
    ├── Melanoma/
    └── NotMelanoma/
```

The names are exact. `calculate_class_weights` opens `training/Melanoma` and
`training/NotMelanoma` by name, so a folder called `benign` or `not_melanoma`
fails there even if `flow_from_directory` was happy.

Images are resized by the loader, so they do not need to arrive at any
particular size. The default is 380 by 380 for EfficientNetB4 and 299 by 299
otherwise.

## Which class is 0

`flow_from_directory` assigns class indices by sorting the folder names, and
`Melanoma` sorts before `NotMelanoma`. So:

| folder | class index |
|---|---|
| `Melanoma` | 0 |
| `NotMelanoma` | 1 |

With `class_mode='binary'` a sigmoid output near **1 means NotMelanoma**, and
near **0 means melanoma**. That is the opposite of what most people assume from
a melanoma classifier, and reading a score the wrong way round turns a confident
correct answer into a confident wrong one. `calculate_class_weights` returns
`{0: weight_melanoma, 1: weight_not_melanoma}`, which matches this ordering.

## Downloading from ISIC

The [ISIC Archive](https://www.isic-archive.com/) is the source. Install the
official client:

```bash
pip install isic-cli
```

The archive is large and heavily imbalanced. As of writing:

| query | images |
|---|---|
| `diagnosis_1:Benign` | 502,877 |
| `diagnosis_1:Malignant` | 27,294 |
| `diagnosis_2:"Malignant melanocytic proliferations (Melanoma)"` | 11,324 |

Downloading everything is roughly half a million images, so filter. Melanoma
first:

```bash
isic image download \
  --search 'diagnosis_2:"Malignant melanocytic proliferations (Melanoma)"' \
  data/raw/Melanoma/
```

Then a comparable set of benign lesions. Take a subset rather than all 502,877,
and keep the ratio deliberate rather than accidental:

```bash
isic image download \
  --search 'diagnosis_1:Benign AND melanocytic:true' \
  data/raw/NotMelanoma/
```

That query is 48,974 images: benign melanocytic lesions, which are the ones a
melanoma classifier has to tell melanoma apart from. Benign non-melanocytic
lesions are an easier negative and will flatter the score.

The metadata is worth pulling too, because it carries the diagnosis chain and,
for some collections, identifiers you need for the next section:

```bash
isic metadata download > data/metadata.csv

# collections are how the challenge datasets are grouped
isic collection list
isic metadata download --collections 70 > data/2020-challenge.csv
```

## Splitting

Do not split the images at random.

The archive contains multiple photographs of the same lesion, and several
lesions from the same patient. If two shots of one mole land on either side of
the train and test boundary, the model can recognise the mole rather than the
melanoma, and the test score measures memory instead of generalisation. That is
one of the ways a number like 94% stops meaning anything, which
[94-percent-accurate.md](94-percent-accurate.md) goes into.

Group first, then split the groups. Where the metadata CSV carries `lesion_id`
or `patient_id`, split on those. Where it does not, be explicit in whatever you
write up that the split is image-level and the number is therefore optimistic.

`ModelConfig.validation_split` is 0.15. A 70/15/15 split of groups is a
reasonable starting point.

## Before you train

```bash
python -c "
from training.utils.data_loader import DataLoader
d = DataLoader('data/')
print(d.get_dataset_info())
print(d.calculate_class_weights())
"
```

If that prints counts for all three directories and a pair of weights, the
layout is right.

## What this does not give you

A test split assembled this way is still drawn from the same archive the model
was trained on, with the same cameras, sites and populations. The open issues
on Fitzpatrick skin type coverage exist because dermoscopic archives
under-represent darker skin, and downloading more of the same archive does not
fix that. Say what your split is made of when you report a number.
