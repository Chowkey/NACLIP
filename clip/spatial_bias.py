"""Content-dependent weighting of NACLIP's Gaussian attention bonus."""
import math

import torch


def validate_guidance(tau, temperature, epsilon):
    if not math.isfinite(tau) or not -1 <= tau <= 1:
        raise ValueError('guidance_tau must be finite and within [-1, 1]')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('guidance_temperature must be finite and positive')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('guidance_epsilon must be finite and positive')


def feature_guided_bias(gaussian, features, tau, temperature, epsilon=1e-6):
    """Return [B, N+1, N+1] bias; features are [N+1, B, D].

    Use raw penultimate-block outputs before the final block's layer norm,
    excluding the CLS token. Compute cosine similarity in float32 even for
    half-precision CLIP. Preserve the baseline's zero CLS row and column.
    Only the fixed Gaussian may be cached; these weights depend on each image.
    """
    validate_guidance(tau, temperature, epsilon)
    patches = features[1:].permute(1, 0, 2).float()
    dot = torch.bmm(patches, patches.transpose(1, 2))
    norms = patches.norm(dim=-1)
    cosine = dot / (norms.unsqueeze(2) * norms.unsqueeze(1) + epsilon)
    gate = torch.sigmoid((cosine.clamp(-1, 1) - tau) / temperature)
    bias = gaussian.expand(features.shape[1], -1, -1).clone()
    bias[:, 1:, 1:] *= gate.to(bias.dtype)
    return bias
