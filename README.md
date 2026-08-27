# Unsupervised Anomaly Detection on MVTec AD — *Bottle*

Detect and localise manufacturing defects on images of bottles while **training only on
defect-free samples**. This is a compact, from-scratch reimplementation of a
[PatchCore](https://arxiv.org/abs/2106.08265)-style pipeline, built as a portfolio project
to show the full workflow: data handling, a frozen-backbone feature extractor, a k-NN
memory bank, anomaly-map post-processing, and a proper image- **and** pixel-level
evaluation.

| | |
|---|---|
| **Task** | One-class / cold-start anomaly detection + segmentation |
| **Dataset** | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), `bottle` category |
| **Backbone** | ImageNet-pretrained **ResNet18**, frozen, truncated after `layer2` |
| **Training** | None — the method is non-parametric (a nearest-neighbour index over normal patches) |
| **Notebook** | [`notebook.ipynb`](notebook.ipynb) — runs end-to-end on a free Colab GPU |

---

## Results

Evaluated on the 83-image `bottle` test set (20 normal, 63 defective across 3 defect types).

| Metric | Value | What it measures |
|---|---|---|
| **Image-level AUROC** | **0.9937** | *"Is this image defective?"* — threshold-free ranking quality |
| Pixel-level IoU | 0.278 | *"Where is the defect?"* — overlap of predicted vs. ground-truth mask |
| Pixel-level Dice | 0.435 | same, F1-style |

Detection is essentially solved on this category; localisation is only rough (see
[Limitations](#limitations)). The notebook also produces:

* a PCA scatter plot of the pooled feature vectors (normal vs. defective),
* histograms of the image-level anomaly scores,
* the ROC curve,
* qualitative panels: *original · ground-truth mask · anomaly heatmap · predicted mask*.

---

## The idea in one paragraph

A network pretrained on ImageNet is already a good generic feature extractor. If we push
every **normal** training image through the first layers of ResNet18, each image becomes a
`28×28` grid of `128`-dimensional **patch descriptors** — small vectors describing the
local appearance of one region. We collect *all* of these normal patch descriptors into a
**memory bank**. To score a new image, we compare each of its patches against the memory
bank: a patch whose nearest normal neighbours are all far away is *unusual* and therefore
likely a defect. The maximum patch score gives a per-image anomaly score; reshaping the
patch scores back onto a grid gives a spatial heatmap we can turn into a segmentation mask.

No defects are ever shown to the model, and nothing is trained by gradient descent.

---

## Method, step by step

### 1. Feature extraction (`Section 3` of the notebook)

ResNet18 is used up to and including `layer2` and then frozen:

| Stage | Output (for a `224×224` input) | Feature character |
|---|---|---|
| `conv1` + `maxpool` | `64 × 56 × 56` | edges, blobs |
| `layer1` | `64 × 56 × 56` | simple textures |
| **`layer2`** ← cut here | **`128 × 28 × 28`** | **mid-level: patterns, corners, local structure** |
| `layer3`, `layer4` | `256 × 14 × 14`, `512 × 7 × 7` | high-level, ImageNet-class-specific |

`layer2` is a deliberate compromise: deep enough that the features are meaningful, shallow
enough that (a) they haven't specialised to ImageNet classes and (b) the `28×28` spatial
resolution is still fine enough to localise a defect.

Each spatial position of the `128 × 28 × 28` map is one patch descriptor, so every image
yields `28 × 28 = 784` descriptors of dimension `128`.

### 2. Memory bank + k-NN scoring (`Section 4`)

* **Build the bank.** Stack the descriptors of all 209 normal training images:
  `209 × 784 ≈ 164 000` rows × `128` columns.
* **Sub-sample.** Randomly keep `50 000` rows to bound memory and query time. *(PatchCore
  uses a greedy coreset selection that preserves coverage better; random sub-sampling is
  the simple stand-in here and is sufficient for a single category.)*
* **Index.** Fit a `scikit-learn` `NearestNeighbors` model on the bank (`k = 9`,
  Euclidean).
* **Score a patch** `p`: let `d_1 … d_k` be the distances to its `k` nearest neighbours in
  the bank. The patch score is their mean, `s(p) = (1/k) Σ d_i`.
* **Score an image:** `S(image) = max over its 784 patches of s(p)` — one strongly
  anomalous region is enough to flag the whole image.

### 3. From patch scores to a mask (`Section 5`)

The `28×28` grid of patch scores is turned into a full-resolution defect mask:

1. **Gaussian smoothing** (`σ = 4`) — removes per-patch blockiness.
2. **Bilinear upsampling** `28×28 → 224×224`.
3. **Min–max normalisation** to `[0, 1]`.
4. **Threshold** at the `95th` percentile, then **morphological** close (fill holes) +
   open (drop specks).

### 4. Evaluation (`Section 6`)

* **Image level:** `roc_auc_score(labels, image_scores)` — AUROC, threshold-free.
* **Pixel level:** IoU and Dice between predicted and ground-truth masks, computed **only
  on defective images** (a normal image has an empty ground-truth mask, so overlap metrics
  are undefined there).

---

## Repository structure

```text
anomaly-detection-mvtec/
├── notebook.ipynb   # the full pipeline, heavily commented, with section-by-section markdown
└── README.md        # this file
```

The notebook is organised into 8 sections:

| # | Section | Content |
|---|---|---|
| 0 | Environment check | GPU / PyTorch version |
| 1 | Setup | imports, seeding, config, dataset stats, sample images |
| 2 | Preprocessing | transforms + the `MVTecDataset` class returning `(image, label, mask)` |
| 3 | Feature extraction | frozen ResNet18, feature dump, PCA sanity check |
| 4 | Memory bank + scoring | build bank, fit k-NN, per-image/per-patch scores, score histograms |
| 5 | Segmentation | smooth → upsample → threshold → morphology; qualitative grid |
| 6 | Evaluation | AUROC, IoU, Dice, ROC curve, normal-vs-defective panel |
| 7 | Summary | results table and concrete next steps |

---

## How to run

### On Google Colab (recommended)

1. Download **MVTec AD** from the
   [official page](https://www.mvtec.com/company/research/datasets/mvtec-ad) (free, for
   research use). Unzip and place the **`bottle`** folder in your Google Drive as
   `MyDrive/bottle`, keeping the original layout:

   ```text
   bottle/
     train/good/*.png                       # defect-free images only
     test/good/*.png                         # defect-free test images
     test/<defect_type>/*.png                # defective test images
     ground_truth/<defect_type>/*_mask.png   # pixel-level masks
   ```

2. Open [`notebook.ipynb`](notebook.ipynb) in Colab (badge at the top of the notebook).
3. **Runtime → Change runtime type → GPU** (a free T4 is enough).
4. Run all cells. Expect a few minutes for feature extraction and k-NN scoring.

### Locally

You need Python 3.10+, a CUDA GPU is optional (CPU works, just slower):

```bash
pip install torch torchvision scikit-learn opencv-python scipy matplotlib tqdm
```

Then remove the `google.colab` drive-mount cell and set `DATASET_ROOT` to your local path
to the `bottle` folder.

### Adapting to another MVTec category

Every step is category-agnostic. Point `DATASET_ROOT` at another category (`cable`,
`hazelnut`, `screw`, …) and re-run. `MAX_PATCHES`, `k`, the `layer2` cut and the `95th`
percentile threshold may need light tuning for texture categories.

---

## Design choices & rationale

| Choice | Why |
|---|---|
| **Frozen ResNet18, no fine-tuning** | There is no labelled defect data to fine-tune on; ImageNet features transfer well and this keeps the method fast and reproducible. |
| **Cut at `layer2`** | Balance between feature abstraction and spatial resolution for localisation. |
| **k-NN memory bank (non-parametric)** | Simple, interpretable, and the reference implementation of the PatchCore family; no training loop to get wrong. |
| **`max` patch aggregation for the image score** | Defects are local — averaging would dilute a small but decisive anomalous region. |
| **Random sub-sampling of the bank** | Coreset selection is the "proper" choice but adds complexity; random is enough for one category and keeps RAM/latency low on free Colab. |
| **Horizontal-flip augmentation on the bank only** | A bottle is left–right symmetric, so a flip yields another valid *normal* view; the test pipeline stays deterministic. |
| **Fixed seeds everywhere** | The two stochastic steps (flip, sub-sampling) become reproducible run-to-run. |

---

## Limitations

* **Pixel-level localisation is coarse.** Two reasons: the anomaly map starts at `28×28`
  resolution, and the segmentation threshold is a *fixed* `95th` percentile — it assumes
  every image has roughly the same defect area and even forces a "defect" region on normal
  images. A threshold calibrated on a validation split, plus threshold-free pixel metrics
  (pixel-AUROC / PRO), would give a fairer and better picture.
* **Single category.** Numbers are reported for `bottle` only; published PatchCore results
  are averaged over all 15 categories.
* **Scoring is slow.** The k-NN queries loop per image and use a `ball_tree`, which
  degrades in 128-D. A brute-force or FAISS index with batched queries would be much
  faster.
* **Random (not coreset) memory bank** — slightly below the full PatchCore in principle.

## Possible next steps

* Greedy **coreset** sub-sampling; concatenate `layer2` **+** `layer3` features.
* **Validation-calibrated** segmentation threshold; report **pixel-AUROC / PRO**.
* **FAISS** index + batched queries for speed.
* Run **all 15 categories** and report the mean.

---

## References

* Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (**PatchCore**),
  CVPR 2022 — [arxiv.org/abs/2106.08265](https://arxiv.org/abs/2106.08265)
* Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly
  Detection*, CVPR 2019 —
  [dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad)
* Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and
  Localization*, ICPR 2021 — [arxiv.org/abs/2011.08785](https://arxiv.org/abs/2011.08785)

## License

Code in this repository: MIT. The MVTec AD dataset is © MVTec Software GmbH and is
released under its own license for **non-commercial research** — see the dataset page.
