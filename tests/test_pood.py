import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from fl_sim.model import build_model
from fl_sim.pood import (
    collect_candidates,
    cosine_knn,
    l2_knn,
    same_label_selection,
    selected_feature_metrics,
    optimize_universal_perturbation,
)


class IdentityFeatureModel(nn.Module):
    def extract_features(self, inputs):
        return inputs.flatten(start_dim=1)


class PoodTests(unittest.TestCase):
    def test_cnn2_extracts_128_dimensional_features(self):
        model = build_model("mnist").eval()
        features = model.extract_features(torch.randn(3, 1, 28, 28))
        self.assertEqual(features.shape, (3, 128))

    def test_collect_candidates_excludes_target_label(self):
        images = torch.randn(8, 1, 28, 28)
        labels = torch.tensor([3, 1, 3, 2, 4, 3, 5, 6])
        dataset = TensorDataset(images, labels)
        selected_images, selected_labels, source_indices = collect_candidates(
            dataset, excluded_label=3, candidate_count=5, seed=1
        )
        self.assertEqual(selected_images.shape, (5, 1, 28, 28))
        self.assertEqual(len(source_indices), 5)
        self.assertNotIn(3, selected_labels.tolist())

    def test_cosine_knn_returns_nearest_features_in_order(self):
        target = torch.tensor([[1.0, 0.0]])
        candidates = torch.tensor([[0.0, 1.0], [0.8, 0.2], [1.0, 0.0]])
        indices, similarities = cosine_knn(target, candidates, k=2)
        self.assertEqual(indices.tolist(), [2, 1])
        self.assertGreater(similarities[0].item(), similarities[1].item())

    def test_l2_knn_returns_smallest_raw_distances_in_order(self):
        target = torch.tensor([[1.0, 0.0]])
        candidates = torch.tensor([[10.0, 0.0], [1.2, 0.1], [0.0, 1.0]])
        indices, distances = l2_knn(target, candidates, k=2)
        self.assertEqual(indices.tolist(), [1, 2])
        self.assertLess(distances[0].item(), distances[1].item())

    def test_same_label_selection_prefers_largest_nearest_group(self):
        label, positions = same_label_selection([8, 2, 8, 4, 8, 2], p=3)
        self.assertEqual(label, 8)
        self.assertEqual(positions, [0, 2, 4])

    def test_selected_feature_metrics(self):
        target = torch.tensor([[1.0, 0.0]])
        selected = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        metrics = selected_feature_metrics(target, selected)
        expected = (1.0 + 2.0**-0.5) / 2.0
        self.assertAlmostEqual(
            metrics["mean_target_feature_similarity"], expected, places=6
        )
        self.assertAlmostEqual(
            metrics["mean_candidate_internal_consistency"], 2.0**-0.5, places=6
        )

    def test_single_selected_feature_has_trivial_internal_consistency(self):
        metrics = selected_feature_metrics(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
        )
        self.assertEqual(metrics["mean_target_feature_similarity"], 0.0)
        self.assertEqual(metrics["mean_candidate_internal_consistency"], 1.0)

    def test_universal_perturbation_reduces_mean_feature_distance(self):
        model = IdentityFeatureModel().eval()
        selected_pixels = torch.tensor([0.1, 0.2]).reshape(2, 1, 1, 1)
        selected = (selected_pixels - 0.1307) / 0.3081
        target_pixel = torch.tensor([0.8]).reshape(1, 1, 1, 1)
        target = model.extract_features((target_pixel - 0.1307) / 0.3081)

        delta, perturbed, _, metrics = optimize_universal_perturbation(
            model=model,
            selected_images=selected,
            target_feature=target,
            device=torch.device("cpu"),
            steps=80,
            learning_rate=0.05,
            epsilon=0.3,
        )

        self.assertEqual(delta.shape, (1, 1, 1, 1))
        self.assertEqual(perturbed.shape, selected.shape)
        self.assertLess(
            metrics["final_mean_feature_distance"],
            metrics["initial_mean_feature_distance"],
        )
        self.assertLessEqual(metrics["perturbation_linf"], 0.300001)


if __name__ == "__main__":
    unittest.main()
