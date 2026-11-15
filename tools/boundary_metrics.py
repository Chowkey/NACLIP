"""Proposal boundary/interior accuracy, independent of the model and MMSeg."""
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


def boundary_band(labels, radius=3, ignore_index=255):
    """Both sides of valid 4-neighbor transitions, Euclidean radius <= radius.

    Operate on the original annotation grid. Ignore interfaces and the image
    border do not create seeds. Distance is geometric, not a geodesic path.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError('labels must be a two-dimensional annotation grid')
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('boundary radius must be finite and nonnegative')
    valid = labels != ignore_index
    seeds = np.zeros(labels.shape, dtype=bool)
    horizontal = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (labels[1:, :] != labels[:-1, :]) & valid[1:, :] & valid[:-1, :]
    seeds[:, 1:] |= horizontal
    seeds[:, :-1] |= horizontal
    seeds[1:, :] |= vertical
    seeds[:-1, :] |= vertical
    if not seeds.any():
        return seeds
    return (distance_transform_edt(~seeds) <= radius) & valid


def accuracy_counts(prediction, labels, radius=3, ignore_index=255):
    prediction, labels = np.asarray(prediction), np.asarray(labels)
    if prediction.shape != labels.shape:
        raise ValueError('prediction and labels must share the original annotation grid')
    band = boundary_band(labels, radius, ignore_index)
    interior = (labels != ignore_index) & ~band
    correct = prediction == labels
    return dict(boundary_correct=int((correct & band).sum()),
                boundary_pixels=int(band.sum()),
                interior_correct=int((correct & interior).sum()),
                interior_pixels=int(interior.sum()))


def summarize_counts(results):
    """Pool pixel counts across images; never average per-image percentages."""
    totals = {key: sum(row[key] for row in results) for key in (
        'boundary_correct', 'boundary_pixels', 'interior_correct', 'interior_pixels')}
    for name in ['boundary', 'interior']:
        count = totals[name + '_pixels']
        totals[name + '_accuracy'] = 100.0 * totals[name + '_correct'] / count if count else float('nan')
    return totals
