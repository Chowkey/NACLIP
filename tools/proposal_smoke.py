"""Exercise all proposal conditions and MMSeg metrics on a synthetic T4 image.

This checks inference/metric integration, not benchmark quality. Run with the
isolated remote model Python after setup; downloads the CLIP checkpoint if needed.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    os.environ['MPLBACKEND'] = 'Agg'
    import torch
    from mmengine.evaluator import Evaluator
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample
    from mmseg.utils import register_all_modules
    import tools.boundary_metric  # noqa: F401
    from naclip import NACLIP

    register_all_modules(init_default_scope=True)
    assert torch.cuda.is_available(), 'Run on a GPU runtime'
    torch.manual_seed(42)
    model = NACLIP(clip_path='ViT-B/16', name_path='configs/cls_voc20.txt',
                   pamr_steps=0).eval().cuda()
    assert model.pamr is None
    assert not any(p.requires_grad for p in model.parameters())
    image = torch.randint(0, 256, (3, 224, 224), dtype=torch.float32)
    labels = torch.zeros(1, 224, 224, dtype=torch.long)
    labels[:, :, 112:] = 1
    labels[:, 0, :] = 255
    records = {}
    with torch.inference_mode():
        for condition in ['fixed', 'guided', 'none']:
            # Test parameters only; these are not calibrated research settings.
            model.net.visual.set_params('reduced', 'naclip', 5., condition, 0.5, 0.1)
            sample = SegDataSample(metainfo=dict(ori_shape=(224, 224), img_shape=(224, 224),
                                                 pad_shape=(224, 224), scale_factor=(1., 1.),
                                                 img_path='synthetic-proposal.png'))
            sample.gt_sem_seg = PixelData(data=labels.clone())
            batch = dict(inputs=[image.clone()], data_samples=[sample])
            outputs = model.test_step(batch)
            assert tuple(outputs[0].pred_sem_seg.data.shape) == (1, 224, 224)
            assert torch.isfinite(outputs[0].seg_logits.data).all()
            evaluator = Evaluator([
                dict(type='IoUMetric', iou_metrics=['mIoU']),
                dict(type='BoundaryAccuracy', radius=3, prefix='localization')])
            evaluator.dataset_meta = dict(classes=tuple(str(i) for i in range(int(model.query_idx.max()) + 1)))
            evaluator.process(data_samples=outputs, data_batch=batch)
            metrics = evaluator.evaluate(1)
            assert metrics['localization/boundary_pixels'] == 8 * 223
            assert metrics['localization/interior_pixels'] == (224 - 8) * 223
            for name in ['boundary_accuracy', 'interior_accuracy']:
                assert 0 <= metrics['localization/' + name] <= 100
            records[condition] = {key: float(value) for key, value in metrics.items()}
            print(f'PROPOSAL_SMOKE {condition}: passed', flush=True)
    Path('/content/proposal-smoke.json').write_text(json.dumps(records, indent=2))
    print('All conditions passed real CLIP inference and MMSeg boundary/IoU integration.')
    print('Synthetic scores are not accuracy measurements. Saved /content/proposal-smoke.json')


if __name__ == '__main__':
    main()
