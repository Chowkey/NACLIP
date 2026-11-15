# NACLIP improvement proposal: feature-guided spatial bias

This note records the proposed research idea and experiment plan from `Proposal.pdf`. It specifies the research hypotheses and comparison protocol. The implementation is exposed through `tools/run_baseline.py --condition fixed|guided|none`; successful execution does not establish an accuracy improvement.

## Core idea and hypothesis

NACLIP adds a distance-based Gaussian bonus to final-layer attention, encouraging nearby patches to share information. Equally distant patches receive the same spatial bonus even when they lie across a semantic class boundary. The key-key attention term can still give those patches different total scores.

**Proposal:** reduce the Gaussian bonus between visually dissimilar patches using features from CLIP's penultimate visual encoder block. This changes the spatial bonus itself while retaining NACLIP's key-key attention.

**Hypothesis:** feature-guided spatial bias improves classification near semantic boundaries while preserving accuracy within regions and overall segmentation quality.

The method remains training-free: freeze CLIP, reuse features from the same forward pass, and add no auxiliary pretrained networks or task-specific parameter training.

## Baseline and controlled comparison

Use NACLIP with frozen CLIP ViT-B/16. Retain its final-block key-key attention and removal of the residual connection and feed-forward network. Use Gaussian standard deviation $\sigma=5$ patch-grid units.

**Disable Pixel-Adaptive Mask Refinement (PAMR) in every condition** to isolate the effect of spatial bias. PAMR-enabled runs are not the baseline specified by this proposal.

Compare three conditions that differ only in the spatial bias added to the final visual attention logits:

| Condition | Added spatial bias |
| --- | --- |
| Fixed Gaussian baseline | Original NACLIP Gaussian $G_{ij}$ |
| Feature-guided proposal | Similarity-weighted Gaussian $B_{ij}^{\mathrm{guided}}$ |
| No spatial bias | Zero; retain key-key attention |

Keep the checkpoint, class names, fixed text template, resizing, normalization, and sliding-window inference settings identical across conditions.

## Proposed method

For patches $i$ and $j$, let $p_i,p_j$ be their patch-grid coordinates and $f_i,f_j$ their features from the **penultimate visual encoder block**.

Compute cosine similarity:

$$
s_{ij} = \frac{f_i^\top f_j}{\lVert f_i\rVert_2\lVert f_j\rVert_2 + \epsilon}.
$$

Compute the original Gaussian bonus:

$$
G_{ij} = \exp\left(-\frac{\lVert p_i-p_j\rVert_2^2}{2\sigma^2}\right).
$$

Weight it by visual similarity:

$$
B_{ij}^{\mathrm{guided}} = G_{ij}\,\operatorname{sigmoid}\left(\frac{s_{ij}-\tau}{T}\right).
$$

- $\epsilon$ provides numerical stability.
- $\tau$ sets the midpoint of similarity weighting.
- $T>0$ controls how gradually the weight changes.
- Lower-similarity pairs retain less of the Gaussian bonus.

Use the same key-key term in every condition, adding $G_{ij}$, $B_{ij}^{\mathrm{guided}}$, or zero before softmax. Set $\tau$ and $T$ before evaluation and share them across both datasets; the proposal does not specify their numerical values.

## Datasets and scoring protocol

| Dataset | Validation images | Evaluated classes |
| --- | ---: | ---: |
| ADE20K-150 | 2,000 | 150 |
| PASCAL Context-59 | 5,105 | 59 |

Exclude background under the PASCAL Context-59 protocol. Resize predictions to the original annotation grid before scoring, and exclude ignored pixels from all metrics.

**Primary metric: boundary accuracy.** Generate a boundary band from each ground-truth mask:

1. Find horizontally or vertically adjacent valid pixels with different labels.
2. Mark both pixels in each such pair as boundary pixels.
3. Include every valid pixel within Euclidean distance **3 pixels** of a marked boundary pixel, on the original annotation grid.
4. Measure the percentage of band pixels whose predicted class matches ground truth.

Use the same ground-truth band for all conditions.

**Overall metric: mIoU.** Calculate per-class intersection over union using all valid pixels, then average across evaluated classes.

Report results separately for each dataset. The main comparison is the **percentage-point change in boundary accuracy** from fixed Gaussian to feature-guided bias, accompanied by overall mIoU and the no-bias results. Inspect region-interior accuracy to check whether boundary gains come at the expense of region consistency.

## Optional experiments

Keep key-key attention and the same inference/scoring settings for both variants:

- **Exponential distance bias:** replace the Gaussian with $B_{ij}=\exp(-r_{ij}/\rho)$, where $r_{ij}=\lVert p_i-p_j\rVert_2$. Choose $\rho$ so the exponential and Gaussian bonuses reach half their maximum at the same distance. Test whether distance-decay shape alone improves localization.
- **Radius-limited attention:** allow attention only to the patch itself and patches within a predefined radius $R$. Retain the original key-key logits inside the radius and set logits outside it to $-\infty$ before softmax. Test whether restricting information exchange improves localization.

## Execution plan and intended contribution

1. Establish the fixed Gaussian and no-bias conditions with **PAMR disabled**; verify boundary-band generation and scoring.
2. Run a pilot on a Colab T4 to measure inference time and GPU memory.
3. Implement feature-guided bias using intermediate features from the same forward pass; verify that only spatial bias differs across the three conditions.
4. Evaluate all three conditions on both full validation splits.
5. Compare boundary accuracy and overall mIoU, inspect representative boundary errors, and run optional variants if time permits.

The intended contribution is a training-free modification that uses visual similarity to suppress inappropriate spatial bonuses across class boundaries. Whether it improves localization while preserving overall quality must be established experimentally.
