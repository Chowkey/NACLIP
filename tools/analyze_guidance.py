"""Label-free penultimate-feature similarity diagnostics for ADE20K and PC59.

Run with the isolated model Python after mounting Drive. This does not measure
segmentation accuracy or establish optimal hyperparameters. All sampled image
IDs, crop limits, seed, and preprocessing are recorded for calibration disclosure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ade-root', type=Path, required=True)
    parser.add_argument('--pc59-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--images-per-dataset', type=int, default=32)
    parser.add_argument('--max-crops-per-image', type=int, default=8)
    parser.add_argument('--pairs-per-image', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    if min(args.images_per_dataset, args.max_crops_per_image, args.pairs_per_image) < 1:
        parser.error('image, crop, and pair limits must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose a new empty output directory')

    os.chdir(ROOT)
    os.environ['MPLBACKEND'] = 'Agg'
    import numpy as np
    import torch
    from mmcv.transforms import Compose
    from mmengine.config import Config
    from mmseg.utils import register_all_modules
    from naclip import NACLIP

    ade_images = sorted((args.ade_root / 'images/validation').glob('*.jpg'))
    split = args.pc59_root / 'ImageSets/SegmentationContext/val.txt'
    if not split.is_file():
        parser.error(f'Missing PC59 validation split: {split}')
    pc_images = sorted(args.pc59_root / 'JPEGImages' / (name.strip()+'.jpg')
                       for name in split.read_text().splitlines() if name.strip())
    if len(ade_images) != 2000 or len(pc_images) != 5105:
        parser.error(f'Expected full image lists (2000, 5105), found ({len(ade_images)}, {len(pc_images)})')
    if not torch.cuda.is_available():
        parser.error('Run with the isolated model Python on the T4')

    register_all_modules(init_default_scope=True)
    model = NACLIP(clip_path='ViT-B/16', name_path='configs/cls_ade20k.txt',
                   pamr_steps=0).eval().cuda()
    capture = []
    handle = model.net.visual.transformer.resblocks[-2].register_forward_hook(
        lambda module, inputs, output: capture.append(output.detach()))
    arrays, records = {}, {}
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    with torch.inference_mode():
        for dataset, paths, cfg_file in [('ade20k', ade_images, 'cfg_ade20k.py'),
                                         ('pc59', pc_images, 'cfg_context59.py')]:
            rng = np.random.default_rng(args.seed)
            chosen = sorted(rng.choice(len(paths), min(args.images_per_dataset, len(paths)), replace=False))
            cfg = Config.fromfile(str(ROOT / 'configs' / cfg_file))
            # Deliberately omit annotations; use exactly the evaluation resize
            # and the model's normalization, then its 224/112 sliding crops.
            pipeline = Compose([step for step in cfg.test_pipeline if step['type'] != 'LoadAnnotations'])
            samples = {'adjacent_r1': [], 'nearby_r5': []}
            crops_seen = 0
            image_records = []
            for index in chosen:
                path = paths[index]
                packed = pipeline(dict(img_path=str(path)))
                prepared = model.data_preprocessor(
                    dict(inputs=[packed['inputs']], data_samples=[packed['data_samples']]), training=False)
                image = prepared['inputs']
                h, w = image.shape[-2:]
                positions = []
                for yi in range(max(h-224+111, 0)//112+1):
                    for xi in range(max(w-224+111, 0)//112+1):
                        y2, x2 = min(yi*112+224, h), min(xi*112+224, w)
                        positions.append((max(y2-224, 0), y2, max(x2-224, 0), x2))
                if len(positions) > args.max_crops_per_image:
                    positions = [positions[i] for i in sorted(rng.choice(
                        len(positions), args.max_crops_per_image, replace=False))]
                image_values = {key: [] for key in samples}
                for y1, y2, x1, x2 in positions:
                    crop = image[:, :, y1:y2, x1:x2]
                    capture.clear()
                    model.net.visual(crop.to(model.dtype), return_all=True)
                    if len(capture) != 1:
                        raise RuntimeError('Expected one penultimate-block capture per crop')
                    features = capture[0][1:, 0, :].float()
                    norms = features.norm(dim=-1)
                    cosine = ((features @ features.T) / (norms[:, None]*norms[None, :]+1e-6)).clamp(-1, 1)
                    gh, gw = crop.shape[-2]//16, crop.shape[-1]//16
                    yy, xx = torch.meshgrid(torch.arange(gh, device=features.device),
                                            torch.arange(gw, device=features.device), indexing='ij')
                    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
                    distance2 = ((coords[:, None]-coords[None, :])**2).sum(-1)
                    upper = torch.triu(torch.ones_like(distance2, dtype=torch.bool), diagonal=1)
                    image_values['adjacent_r1'].append(cosine[upper & (distance2 == 1)].cpu().numpy())
                    image_values['nearby_r5'].append(cosine[upper & (distance2 <= 25)].cpu().numpy())
                    crops_seen += 1
                image_records.append(dict(image_id=path.stem, crops=len(positions),
                                          coordinates_y1_y2_x1_x2=positions))
                for key in samples:
                    values = np.concatenate(image_values[key])
                    # Equal-sized samples per image prevent wide images with
                    # more sliding windows dominating the pooled distribution.
                    selected = rng.choice(values.size, args.pairs_per_image,
                                          replace=values.size < args.pairs_per_image)
                    samples[key].append(values[selected])
                if len(image_records) % 4 == 0:
                    print(f'DISTRIBUTION {dataset} {len(image_records)}/{len(chosen)} images', flush=True)
            distributions = {}
            for key, values in samples.items():
                values = np.concatenate(values)
                arrays[dataset+'_'+key] = values
                distributions[key] = dict(mean=float(values.mean()), std=float(values.std()),
                    quantiles={f'p{q:02d}':float(v) for q,v in zip(percentiles,np.percentile(values,percentiles))},
                    resampled_pairs=int(values.size))
            records[dataset] = dict(images=image_records, crops=crops_seen, distributions=distributions)
            print(dataset, json.dumps(distributions), flush=True)
    handle.remove()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(purpose='Unlabeled input-distribution calibration; no masks or accuracy scores used',
                  calibration_split='validation images only; no segmentation labels accessed',
                  backbone='CLIP ViT-B/16', gaussian_std=5., torch_version=torch.__version__,
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                                 ['clip/model.py','clip/spatial_bias.py','naclip.py','tools/analyze_guidance.py',
                                  'configs/cfg_ade20k.py','configs/cfg_context59.py']},
                  feature_source='raw penultimate block output before final-block LN; CLS excluded',
                  neighborhoods='unique off-diagonal pairs: r=1 and 0<r<=5 patch units; sigma=5',
                  equal_image_weighting=True, seed=args.seed, images_per_dataset=args.images_per_dataset,
                  max_crops_per_image=args.max_crops_per_image, pairs_per_image=args.pairs_per_image,
                  data_roots=dict(ade20k=str(args.ade_root),pc59=str(args.pc59_root)), datasets=records)
    (args.output_dir/'distribution.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output_dir/'similarities.npz', **arrays)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for dataset, color in [('ade20k','#2563eb'),('pc59','#dc2626')]:
        ax.hist(arrays[dataset+'_nearby_r5'], bins=np.linspace(-1,1,101),
                density=True, histtype='step', linewidth=2, label=dataset, color=color)
    ax.set(xlabel='Penultimate-feature cosine similarity (0 < distance <= 5 patches)',
           ylabel='Density', title='ADE20K vs PC59: equal image weighting, no labels used')
    ax.legend(); fig.tight_layout();fig.savefig(args.output_dir/'similarity_distribution.png',dpi=160)
    print('Saved distribution report, samples, and plot to',args.output_dir,flush=True)


if __name__ == '__main__':
    main()
