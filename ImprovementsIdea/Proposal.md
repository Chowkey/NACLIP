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

## Additional exploratory direction: overlap-aware window calibration

This is implemented as an optional output-stage mode in `naclip.py` and `tools/run_baseline.py`; it is off by default. NACLIP processes overlapping crops independently and normally averages their logits. The new mode also measures how much their class distributions disagree at each pixel, then adjusts only those pixels toward a confidence-weighted crop consensus.

For each image pixel $x$, let $z_w(x)$ be the logits from crop $w$ covering it, and $p_w(x)=\operatorname{softmax}(s z_w(x))$, where $s$ is NACLIP's configured logit scale (40 in the pilot). The standard merged prediction is $p_0(x)=\operatorname{softmax}(s\frac{1}{|W_x|}\sum_w z_w(x))$. Define the unweighted probability mean $\bar p(x)=\frac{1}{|W_x|}\sum_w p_w(x)$ and normalized disagreement

$$
d(x)=\operatorname{clip}_{[0,1]}\left(\frac{H(\bar p(x))-\frac{1}{|W_x|}\sum_w H(p_w(x))}{\min(\log |W_x|,\log C)}\right),
$$

where $H$ is categorical entropy and $C$ is the number of text queries. This is normalized multi-crop Jensen-Shannon divergence. Assign each crop confidence $q_w(x)=\max(1-H(p_w(x))/\log C,\epsilon)^\gamma$ and form $p_c(x)=\frac{\sum_w q_w(x)p_w(x)}{\sum_w q_w(x)}$. The output is

$$
p_{\mathrm{out}}(x)=(1-\lambda d(x))p_0(x)+\lambda d(x)p_c(x),
$$

with default $\lambda=0.5$ and $\gamma=1$. One-crop pixels have zero disagreement and remain equal to the standard merged prediction. The fixed baseline path is unchanged when this mode is disabled. It requires PAMR off in the controlled comparison.

**Relation to prior work and novelty status:** GLA-CLIP identifies the same general problem—semantic discrepancies between independently processed sliding windows—and addresses it inside attention by extending key/value context across windows, adding proxy anchors, and applying scale-aware normalization ([Lee et al., CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Lee_Looking_Beyond_the_Window_Global-Local_Aligned_CLIP_for_Training-free_Open-Vocabulary_CVPR_2026_paper.pdf)). This implementation instead reuses predictions from the existing crops after each crop has run; it adds no global cross-window attention or vision encoder. The mechanism and resource cost differ, but this is not by itself evidence of novelty; related output-fusion methods must be checked before making a novelty claim.

**Pilot protocol:** compare ADE20K fixed Gaussian with and without overlap calibration on the same 32-image subset, seed, and preprocessing. Report matched image IDs, mIoU, boundary/interior accuracy, validity, and runtime. Only consider a full-split comparison if the pilot is valid and shows a repeatable gain without meaningful regression. Keep PAMR disabled.

## Additional candidate: patch-grid phase marginalization

The current sliding-window inference always presents each crop to ViT-B/16 with the same 16-pixel patch-grid alignment. As a result, a small translation of the crop origin can change which pixels are grouped into patch tokens, potentially changing predictions around small objects and class boundaries. Test whether averaging predictions from a small set of shifted patch-grid phases reduces this sensitivity.

For a phase offset $(\delta_x,\delta_y)$, pad/shift the input image by that offset, run the unchanged NACLIP sliding-window inference, then inverse-align the output logits to the original image coordinates. Average aligned logits over the canonical phase and three half-patch phases:

$$
\mathcal{P}=\{(0,0),(8,0),(0,8),(8,8)\},\qquad
z_{\mathrm{phase}}(x)=\frac{1}{|\mathcal{P}|}\sum_{\delta\in\mathcal{P}}z_\delta(x).
$$

Use reflection padding at image edges (replication when the image is too small) and crop back to the original extent after inverse alignment. Pad the far edges so the final sliding crop remains on the same regular stride grid as the preceding crops. Do not change the text prompts, spatial-bias condition, image scale, or other inference settings. The local implementation exposes `--patch-phase-ensemble off|two|four`; `two` averages the canonical and diagonal half-patch phase, while `four` uses all listed phases. This is a test-time ensemble of the same frozen model; it requires two or four image-encoder passes and is expected to increase inference time accordingly. The Colab notebook includes a paired 32-image ADE20K pilot for single-phase versus two-phase inference. A two-phase pilot can first check whether any stability/accuracy signal justifies the full four-phase run.

**Prior-work check and novelty status:** Ercan's 2026 preprint *Phase Marginalization for Patch-Grid Instability in Vision Transformers* studies phase averaging for dense prediction and reports segmentation experiments, so phase marginalization itself is not a new claim. The potential project contribution is a controlled transfer/evaluation on training-free open-vocabulary NACLIP, including its overlapping-crop inference and boundary metrics. This is an adaptation hypothesis, not an established improvement; related preprints and publication status should be rechecked before making a novelty claim ([paper](https://arxiv.org/abs/2606.08132)). NACLIP's own method defines the relevant training-free, patch-attention baseline ([WACV 2025 paper](https://openaccess.thecvf.com/content/WACV2025/papers/Hajimiri_Pay_Attention_to_Your_Neighbours_Training-Free_Open-Vocabulary_Semantic_Segmentation_WACV_2025_paper.pdf)).

**Pilot protocol:** on a fixed, seeded ADE20K subset, compare the existing single-phase baseline against two phases first, then four if useful. Keep PAMR and the chosen spatial-bias mode identical. Verify identical image IDs, inverse alignment, and valid pixels; report mIoU, boundary accuracy, interior accuracy, prediction disagreement across phases, wall time, and peak memory. Advance to the full validation split only if the pilot improves boundary accuracy without a meaningful mIoU regression and the extra inference cost is acceptable.

## Execution plan and intended contribution

1. Establish the fixed Gaussian and no-bias conditions with **PAMR disabled**; verify boundary-band generation and scoring.
2. Run a pilot on a Colab T4 to measure inference time and GPU memory.
3. Implement feature-guided bias using intermediate features from the same forward pass; verify that only spatial bias differs across the three conditions.
4. Evaluate all three conditions on both full validation splits.
5. Compare boundary accuracy and overall mIoU, inspect representative boundary errors, and run optional variants if time permits.

The intended contribution is a training-free modification that uses visual similarity to suppress inappropriate spatial bonuses across class boundaries. Whether it improves localization while preserving overall quality must be established experimentally.
