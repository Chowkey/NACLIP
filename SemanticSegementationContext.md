# Open-Vocabulary Semantic Segmentation: Context and Research Focus

Updated: 2026-11-07. This note summarizes the research setting and a working direction for this NACLIP-based project. The priorities below are research proposals inferred from choosing a training-free baseline, not completed experiments or established contributions.

## 1. What problem are we solving?

Semantic segmentation assigns a semantic category to every pixel. Conventional systems learn a fixed label set; adding categories usually requires new annotations and adaptation. Open-vocabulary semantic segmentation (OVSS) instead uses text descriptions to specify categories at inference time, aiming to extend recognition beyond the segmentation labels used during adaptation.

CLIP makes this possible by learning a shared image-text representation from large-scale image-caption pairs. Its original objective supports image-level recognition; it does not directly teach accurate pixel masks. The central challenge is therefore to convert broad semantic knowledge into spatially precise predictions. See the [CLIP paper](https://proceedings.mlr.press/v139/radford21a.html).

“Open vocabulary” still requires a candidate vocabulary for a particular inference run. It does not imply exhaustive concept discovery or reliable rejection of everything outside that vocabulary. Likewise, an evaluation category absent from segmentation adaptation may have appeared in foundation-model pretraining.

## 2. Where training-free methods fit

OVSS methods differ in their adaptation resources: segmentation supervision, additional image-text training, unlabeled-data adaptation, or inference-only reuse of pretrained models. Comparisons must disclose these differences.

For this project, the intended starting constraint is **frozen CLIP, no additional parameter optimization, no extra adaptation dataset, and no auxiliary pretrained vision model**. Existing foundation-model pretraining remains a dependency. Ground-truth masks are used for evaluation, with any development-set decisions disclosed separately.

Training-free does not automatically mean data-free, auxiliary-model-free, or inexpensive. For example, [FLOSS](https://openaccess.thecvf.com/content/ICCV2025/html/Benigmim_FLOSS_Free_Lunch_in_Open-vocabulary_Semantic_Segmentation_ICCV_2025_paper.html) selects text-template experts using unlabeled images, while [NERVE](https://arxiv.org/abs/2511.08248) uses Stable Diffusion attention. These are useful research references, but their resource assumptions differ from our starting constraint.

## 3. Gaps the field still faces

The following connects published observations to questions we should test; it is not a claim that every method fails in the same way.

| Gap | Research implication |
| --- | --- |
| Image-level alignment does not guarantee spatial alignment. | A model can recognize an object while spreading its evidence into unrelated regions. [SCLIP](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03178.pdf) studies location misalignment and changes attention for dense prediction. |
| Spatial consistency can conflict with boundaries. | A fixed neighborhood encourages coherent regions but may mix adjacent categories. Test thin structures, small objects, and boundary leakage. [NERVE](https://arxiv.org/abs/2511.08248) explicitly examines limitations of fixed isotropic neighborhoods. |
| Useful spatial information extends beyond the final layer. | [ResCLIP](https://arxiv.org/abs/2411.15851) exploits intermediate-layer attention. We should investigate layer/head selection without assuming all internal features are equally helpful. |
| Text classifiers affect pixel predictions. | Class wording, synonyms, and templates can change class competition. [FLOSS](https://openaccess.thecvf.com/content/ICCV2025/html/Benigmim_FLOSS_Free_Lunch_in_Open-vocabulary_Semantic_Segmentation_ICCV_2025_paper.html) shows that averaged templates can underperform class-specific alternatives. |
| Benchmark scores incompletely describe deployment behavior. | As project evaluation questions, test vocabulary changes, background handling, domain changes, latency, and memory alongside mIoU. A higher average score alone cannot establish robustness. |

## 4. Why choose NACLIP as the baseline?

Neighbour-Aware CLIP (NACLIP), introduced in WACV 2025, adapts CLIP for dense prediction by introducing explicit neighborhood consistency. It offers an interpretable starting point without additional training data or auxiliary pretrained networks. It is a baseline for controlled investigation, not an assertion of current state-of-the-art performance. See the [NACLIP paper](https://arxiv.org/html/2404.08181v2).

In this checkout:

- `clip/model.py` uses key-key similarity plus an additive Gaussian spatial bias before softmax for `attn_strategy='naclip'`.
- The `reduced` architecture omits the final block's residual additions and MLP. Earlier transformer blocks retain their normal computation.
- `naclip.py` compares normalized patch features with normalized, template-averaged text embeddings, then upsamples the scores. Its optional PAMR refinement is a separate inference component.
- The default backbone is `ViT-B/16`; the main reproduction setting is `reduced`, `naclip`, Gaussian standard deviation `5`, and PAMR enabled. Dataset configurations also control resizing, sliding windows, class names, and thresholds.

These components expose useful ablations: conventional attention, key-key attention, neighborhood-only attention, reduced architecture, and refinement. Their separation helps explain where an improvement comes from.

## 5. What are we focusing on?

**Working research question: Can we improve localization and region consistency while preserving CLIP's text alignment, keeping weights frozen, and adding little inference cost?**

Our first focus should be understanding NACLIP's remaining failures. Classify errors into incorrect semantics, fragmented regions, boundary leakage, missed small objects, and background confusion. This determines whether to investigate spatial aggregation, internal features, or text scoring.

The most direct hypothesis is that content-sensitive neighborhood aggregation could preserve boundaries better than a fixed Gaussian prior. A second hypothesis is that selective reuse of intermediate attention could recover spatial detail. Both require controlled tests; adaptive neighborhoods and intermediate features already appear in prior work, so neither idea alone establishes novelty.

Text wording and confidence/background handling should initially be controlled variables. If they become the research focus, evaluate them separately so their effects are distinguishable from visual-feature changes. Any use of unlabeled calibration images or additional models must be reported as a changed resource setting.

## 6. What would count as convincing progress?

1. **Reproduce before modifying.** Establish a dataset baseline with the repository settings and record the commit, environment, checkpoint, split, vocabulary, preprocessing, and refinement settings.
2. **Isolate the mechanism.** Change one component at a time; report PAMR on/off separately and match the backbone, image sizes, and inference settings across comparisons.
3. **Measure quality and cost.** Report mIoU, per-class IoU, boundary quality, relevant failure examples, inference time, and peak GPU memory on matched hardware. Inspect small/thin-object subsets where annotations permit.
4. **Test transfer.** Start with a reproducible benchmark such as VOC21, then include a different scene distribution such as ADE20K or Cityscapes. Keep VOC20/VOC21 and other background protocols distinct. Avoid tuning on reported test sets; disclose all validation-based selection.
5. **Keep claims proportional to evidence.** Compare with NACLIP and relevant later methods under explicit resource assumptions. Record regressions and overhead as well as gains.

The [README](README.md) contains upstream reference scores, not measurements reproduced in this fork. The [Colab workflow](COLAB.md) records a successful synthetic T4 inference check on 2026-11-07; it explicitly records no dataset benchmark. Follow that workflow for GPU evaluation and retain experiment outputs outside Git. This document adds research context only and establishes no new accuracy results.
