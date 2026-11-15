"""Proposal correctness checks. Torch checks run in the remote model environment."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.boundary_metrics import accuracy_counts, boundary_band, summarize_counts

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from clip.model import VisionTransformer, convert_weights
    from clip.spatial_bias import feature_guided_bias


class BoundaryTests(unittest.TestCase):
    def test_band_matches_brute_force_euclidean_distance(self):
        labels = np.zeros((11, 13), dtype=np.uint8)
        labels[5, 6] = 1
        labels[0, 0] = 255
        seeds = [(4, 6), (5, 5), (5, 6), (5, 7), (6, 6)]
        expected = np.array([[min((y-a)**2 + (x-b)**2 for a, b in seeds) <= 9
                              for x in range(13)] for y in range(11)])
        expected[0, 0] = False
        np.testing.assert_array_equal(boundary_band(labels), expected)
        self.assertFalse(expected[2, 3])  # Chebyshev/square dilation would include this.

    def test_both_sides_of_transition_are_seeds(self):
        labels = np.array([[0, 0, 1, 1]])
        np.testing.assert_array_equal(boundary_band(labels, radius=0), [[False, True, True, False]])

    def test_ignore_border_and_diagonal_do_not_create_boundaries(self):
        for labels in [np.array([[0, 255, 0]]), np.array([[0, 255], [255, 1]]),
                       np.zeros((3, 3), dtype=int), np.array([[255]]), np.array([[0]])]:
            self.assertFalse(boundary_band(labels).any())

    def test_accuracy_excludes_ignored_and_separates_interior(self):
        labels = np.array([[0, 0, 1, 1, 1, 255]])
        pred = np.array([[0, 1, 1, 0, 1, 0]])
        self.assertEqual(accuracy_counts(pred, labels, radius=0),
                         dict(boundary_correct=1, boundary_pixels=2,
                              interior_correct=2, interior_pixels=3))

    def test_pool_pixel_counts_not_image_percentages(self):
        rows = [dict(boundary_correct=1, boundary_pixels=n, interior_correct=0, interior_pixels=0)
                for n in [1, 9]]
        result = summarize_counts(rows)
        self.assertEqual(result['boundary_accuracy'], 20.)
        self.assertTrue(np.isnan(result['interior_accuracy']))

    def test_invalid_shapes_and_radius_are_rejected(self):
        with self.assertRaises(ValueError):
            accuracy_counts(np.zeros((2, 2)), np.zeros((3, 3)))
        for radius in [-1, float('nan')]:
            with self.assertRaises(ValueError):
                boundary_band(np.zeros((2, 2)), radius)


@unittest.skipUnless(HAS_TORCH, 'Run attention tests with the isolated Colab model Python')
class SpatialBiasTests(unittest.TestCase):
    def make_model(self, condition='fixed'):
        torch.manual_seed(42)
        model = VisionTransformer(4, 2, 16, 3, 2, 8).eval()
        model.set_params('reduced', 'naclip', 5., condition, 0.5, 0.1)
        return model

    def test_guidance_formula_zero_vectors_batch_and_cls(self):
        features = torch.tensor([[[100., 0.], [0., 100.]],
                                 [[1., 0.], [1., 0.]],
                                 [[1., 0.], [-1., 0.]],
                                 [[0., 0.], [0., 0.]]])
        gaussian = torch.ones(1, 4, 4)
        gaussian[:, 0, :] = gaussian[:, :, 0] = 0
        bias = feature_guided_bias(gaussian, features, 0.5, 0.1)
        self.assertTrue(torch.isfinite(bias).all())
        self.assertEqual(tuple(bias.shape), (2, 4, 4))
        self.assertTrue(torch.equal(bias[:, 0, :], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(bias[:, :, 0], torch.zeros(2, 4)))
        self.assertAlmostEqual(bias[0, 1, 2].item(), torch.sigmoid(torch.tensor(5.)).item(), places=5)
        self.assertAlmostEqual(bias[1, 1, 2].item(), torch.sigmoid(torch.tensor(-15.)).item(), places=5)
        self.assertTrue(((bias >= 0) & (bias <= gaussian)).all())

    def test_all_conditions_match_explicit_attention_formula(self):
        model = self.make_model()
        x, features = torch.randn(7, 2, 16), torch.randn(7, 2, 16)
        layer = model.transformer.resblocks[-1].attn
        _, k, _ = F.linear(x, layer.in_proj_weight, layer.in_proj_bias).chunk(3, -1)
        k = k.contiguous().view(7, 4, 8).transpose(0, 1)
        logits = torch.bmm(k, k.transpose(1, 2)) * 8**-0.5
        gaussian = model.get_attention_addition(2, 3, model.gaussian_window(3, 5, std=5.)).unsqueeze(0)
        for condition in ['fixed', 'guided', 'none']:
            model.set_params('reduced', 'naclip', 5., condition, 0.5, 0.1)
            actual = model.custom_attn(layer, x, (2, 3), return_attn=True, guidance_features=features)
            addition = 0 if condition == 'none' else gaussian
            if condition == 'guided':
                addition = feature_guided_bias(gaussian, features, 0.5, 0.1).repeat_interleave(2, dim=0)
            torch.testing.assert_close(actual, (logits + addition).softmax(-1))

    def test_forward_uses_raw_penultimate_features_and_same_pass(self):
        model = self.make_model('guided')
        captured = []
        handle = model.transformer.resblocks[-2].register_forward_hook(
            lambda module, args, output: captured.append(output.clone()))
        try:
            with torch.inference_mode(), patch('clip.model.feature_guided_bias', wraps=feature_guided_bias) as guided:
                output = model(torch.randn(2, 3, 4, 6), return_all=True)
                self.assertEqual(tuple(output.shape), (2, 7, 8))
                self.assertEqual(len(captured), 1)
                torch.testing.assert_close(guided.call_args.args[1], captured[0])
        finally:
            handle.remove()

    def test_batch_equals_independent_images_and_gate_is_not_cached(self):
        model = self.make_model('guided')
        images = torch.randn(2, 3, 4, 6)
        with torch.inference_mode():
            batch = model(images, return_all=True)
            separate = torch.cat([model(image[None], return_all=True) for image in images])
        torch.testing.assert_close(batch, separate, atol=2e-6, rtol=2e-5)
        self.assertFalse(torch.equal(batch[0], batch[1]))

    def test_validation_and_cache_invalidation(self):
        model = self.make_model()
        with torch.inference_mode():
            model(torch.randn(1, 3, 4, 4))
        self.assertTrue(model.addition_cache)
        model.set_params('reduced', 'naclip', 2.)
        self.assertFalse(model.addition_cache)
        for tau, temperature in [(2., 1.), (0., 0.), (0., float('nan'))]:
            with self.assertRaises(ValueError):
                model.set_params('reduced', 'naclip', 5., 'guided', tau, temperature)

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available() if HAS_TORCH else False, 'CUDA required')
    def test_cuda_half_precision_guided_forward(self):
        model = self.make_model('guided').cuda()
        # Match CLIP loading: projection/attention weights are fp16, while
        # layer-norm weights remain fp32. model.half() would break CLIP's LN.
        convert_weights(model)
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 4, 6, device='cuda').half(), return_all=True)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
