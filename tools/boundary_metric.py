"""MMSeg adapter for the proposal's original-resolution boundary accuracy."""
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

from tools.boundary_metrics import accuracy_counts, summarize_counts


@METRICS.register_module()
class BoundaryAccuracy(BaseMetric):
    default_prefix = 'boundary'

    def __init__(self, radius=3, ignore_index=255, collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.radius = radius
        self.ignore_index = ignore_index

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            prediction = sample['pred_sem_seg']['data'].detach().cpu().numpy()[0]
            labels = sample['gt_sem_seg']['data'].detach().cpu().numpy()[0]
            self.results.append(accuracy_counts(prediction, labels, self.radius, self.ignore_index))

    def compute_metrics(self, results):
        # Counts are returned too, so empty bands remain distinguishable from
        # zero accuracy. BaseMetric handles distributed gathering before this.
        return summarize_counts(results)
