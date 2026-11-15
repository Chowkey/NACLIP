"""Run the unchanged NACLIP configuration with reproducible subsets and artifacts.

Use the isolated Colab model Python. Prediction export uses a hook rather than
eval.py's visualization option, which changes the model's PAMR configuration.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['ade20k', 'pc59'], required=True)
    parser.add_argument('--stage', choices=['subset', 'full'], required=True)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--subset-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.subset_size < 2:
        parser.error('--subset-size must be at least 2 for timing statistics')
    if not args.data_root.is_dir():
        parser.error(f'Missing dataset: {args.data_root}')
    if str(args.work_dir).startswith('/content/drive/') and not Path('/content/drive/MyDrive').is_dir():
        parser.error('Mount Drive before saving persistent results')
    if (args.work_dir / 'summary.json').exists():
        parser.error('This run already has a summary; use a new work directory')
    os.chdir(ROOT)
    # Colab exports a notebook-only backend absent in the isolated environment.
    os.environ['MPLBACKEND'] = 'Agg'

    import numpy as np
    from PIL import Image
    import torch
    from mmengine.config import Config
    from mmengine.hooks import Hook
    from mmengine.runner import Runner
    from mmseg.registry import DATASETS
    from mmseg.utils import register_all_modules
    import custom_datasets  # noqa: F401; registers datasets
    import naclip  # noqa: F401; registers the unchanged model

    assert torch.cuda.is_available(), 'GPU runtime required'
    register_all_modules(init_default_scope=True)
    config = 'configs/cfg_ade20k.py' if args.dataset == 'ade20k' else 'configs/cfg_context59.py'
    cfg = Config.fromfile(str(ROOT / config))
    cfg.work_dir = str(args.work_dir)
    cfg.randomness = dict(seed=args.seed)
    cfg.test_dataloader.dataset.data_root = str(args.data_root)
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    # Do not call eval.visualization_hook: it silently changes PAMR settings.
    cfg.default_hooks.pop('visualization', None)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    dataset.full_init()
    expected = 2000 if args.dataset == 'ade20k' else 5105
    assert len(dataset) == expected, f'Expected {expected} validation images, found {len(dataset)}'
    ordered = sorted((Path(dataset.get_data_info(i)['img_path']).stem, i) for i in range(len(dataset)))
    if args.stage == 'subset':
        positions = sorted(np.random.default_rng(args.seed).choice(len(ordered), min(args.subset_size, len(ordered)), replace=False).tolist())
        selected = [ordered[p] for p in positions]
        cfg.test_dataloader.dataset.indices = [index for _, index in selected]
    else:
        selected = ordered
    write_json(args.work_dir / 'image_ids.json', [name for name, _ in selected])
    cfg.dump(str(args.work_dir / 'config.py'))

    source_files = ['naclip.py', 'clip/model.py', 'clip/clip.py', 'pamr.py',
                    'custom_datasets.py', config, 'configs/base_config.py',
                    'prompts/imagenet_template.py', 'tools/run_baseline.py']
    source_files += ['configs/cls_ade20k.txt' if args.dataset == 'ade20k' else 'configs/cls_context59.txt']
    versions = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True) if subprocess.run([sys.executable, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 else ''
    # uv-created environments may not contain pip; metadata needs no installer.
    if not versions:
        from importlib.metadata import distributions
        versions = '\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in distributions()))
    (args.work_dir / 'packages.txt').write_text(versions)
    manifest = dict(dataset=args.dataset, stage=args.stage, image_count=len(selected),
                    seed=args.seed, config=config, data_root=str(args.data_root),
                    python=platform.python_version(), torch=torch.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
                    timing_definition='CUDA-synchronized test iteration including preprocessing, inference and metric processing; excludes dataloader fetch and prediction export',
                    accuracy_note='Subset mIoU is a development measurement; only full-split mIoU is a baseline result.')
    write_json(args.work_dir / 'manifest.json', manifest)
    timings = []
    pamr_failures = []

    class RecordPAMRFailures(logging.Handler):
        def emit(self, record):
            if "Couldn't apply PAMR" in record.getMessage():
                pamr_failures.append(record.getMessage())

    class RecordPredictions(Hook):
        def before_test_iter(self, runner, batch_idx, data_batch=None):
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            torch.cuda.synchronize()
            seconds = time.perf_counter() - self.started
            for sample in outputs:
                image_id = Path(sample.img_path).stem
                timings.append(dict(image_id=image_id, test_iteration_seconds=seconds))
                if args.stage == 'subset':
                    prediction = sample.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
                    folder = args.work_dir / 'predictions'
                    folder.mkdir(exist_ok=True)
                    Image.fromarray(prediction).save(folder / f'{image_id}.png')
                    palette = np.asarray(runner.test_dataloader.dataset.metainfo['palette'], dtype=np.uint8)
                    with Image.open(sample.img_path) as image:
                        image = image.convert('RGB')
                        rgb = np.asarray(image)
                    colored = palette[prediction]
                    overlay = Image.fromarray((0.5 * rgb + 0.5 * colored).astype(np.uint8))
                    gt = sample.gt_sem_seg.data.squeeze(0).cpu().numpy()
                    gt_color = np.zeros((*gt.shape, 3), dtype=np.uint8)
                    valid = gt != 255
                    gt_color[valid] = palette[gt[valid]]
                    gt_overlay = Image.fromarray((0.5 * rgb + 0.5 * gt_color).astype(np.uint8))
                    panel = Image.new('RGB', (image.width * 3, image.height))
                    for i, tile in enumerate([image, gt_overlay, overlay]):
                        panel.paste(tile, (i * image.width, 0))
                    views = args.work_dir / 'comparisons'
                    views.mkdir(exist_ok=True)
                    panel.save(views / f'{image_id}.jpg', quality=92)
            if (batch_idx + 1) % 25 == 0 or args.stage == 'subset':
                write_json(args.work_dir / 'progress.json', dict(completed=len(timings), total=len(selected),
                           elapsed_seconds=time.perf_counter() - test_started,
                           pamr_failures=pamr_failures))
                print(f'BASELINE_PROGRESS {args.dataset} {args.stage} {len(timings)}/{len(selected)}', flush=True)

    capture = RecordPAMRFailures()
    logging.getLogger().addHandler(capture)
    initialized = time.perf_counter()
    runner = Runner.from_cfg(cfg)
    init_seconds = time.perf_counter() - initialized
    runner.register_hook(RecordPredictions(), priority='LOWEST')
    model = runner.model
    manifest['resolved_model'] = {name: getattr(model, name) for name in ['logit_scale', 'prob_thd', 'slide_stride', 'slide_crop']}
    manifest['resolved_model'].update(arch=model.net.visual.arch, attn_strategy=model.net.visual.attn_strategy,
                                      gaussian_std=model.net.visual.gaussian_std,
                                      pamr_steps=model.pamr.num_iter if model.pamr is not None else 0,
                                      pamr_dilations=[8, 16])
    write_json(args.work_dir / 'manifest.json', manifest)
    write_json(args.work_dir / 'classes.json', runner.test_dataloader.dataset.metainfo['classes'])
    torch.cuda.reset_peak_memory_stats()
    test_started = time.perf_counter()
    metrics = runner.test()
    elapsed = time.perf_counter() - test_started
    write_json(args.work_dir / 'timing.json', timings)
    # First sample is retained in all metrics, but excluded from steady-state latency.
    steady = [row['test_iteration_seconds'] for row in timings[1:]]
    assert len(timings) == len(selected), 'Evaluation did not visit every selected image'
    summary = dict(dataset=args.dataset, stage=args.stage, images=len(timings),
                   metrics={k: float(v) for k, v in metrics.items()},
                   initialization_seconds=init_seconds, test_wall_seconds=elapsed,
                   steady_mean_seconds=float(np.mean(steady)), steady_median_seconds=float(np.median(steady)),
                   peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   pamr_failures=pamr_failures, baseline_valid=not pamr_failures,
                   comparison_layout='left: image; middle: ground-truth overlay; right: prediction overlay')
    write_json(args.work_dir / 'summary.json', summary)
    print('BASELINE_SUMMARY ' + json.dumps(summary), flush=True)
    if pamr_failures:
        raise RuntimeError('PAMR failed on some images; this is not a valid unchanged baseline')


if __name__ == '__main__':
    main()
