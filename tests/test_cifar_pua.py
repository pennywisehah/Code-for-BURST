import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from fl_sim.CIFAR_PUA import (
    DEFAULT_DIRICHLET_ALPHA,
    build_parser,
    choose_cifar10_targets,
    distribute_poison_record_positions,
    evaluate_targets,
    latest_cifar10_checkpoint,
    parse_target_indices,
    parse_client_ids,
    resolve_malicious_client_ids,
    resolve_pood_training_label,
    summarize_target_group,
)
from fl_sim.cifar_pood import (
    add_cifar10_pixel_perturbation,
    denormalize_cifar10,
    load_cifar10_proxy,
    select_cifar100_pood,
)
from fl_sim.model import build_model, extract_model_features


class CifarPoodTests(unittest.TestCase):
    def test_multiple_malicious_clients_split_one_total_budget_evenly(self):
        allocation = distribute_poison_record_positions(60, [4, 7])
        self.assertEqual(len(allocation[4]), 30)
        self.assertEqual(len(allocation[7]), 30)
        self.assertEqual(
            sorted(allocation[4] + allocation[7]), list(range(60))
        )
        self.assertFalse(set(allocation[4]) & set(allocation[7]))

    def test_plural_malicious_client_option_overrides_legacy_single_option(self):
        args = build_parser().parse_args(
            ["--malicious-client-id", "6", "--malicious-client-ids", "4,7"]
        )
        self.assertEqual(parse_client_ids("4,7"), [4, 7])
        self.assertEqual(resolve_malicious_client_ids(args), [4, 7])
        with self.assertRaisesRegex(Exception, "unique"):
            parse_client_ids("4,4")

    def test_group_evaluation_returns_one_record_per_target(self):
        model = build_model("cifar10")
        evaluations = evaluate_targets(
            model.state_dict(),
            torch.randn(3, 3, 32, 32),
            torch.tensor([1, 2, 3]),
            torch.device("cpu"),
        )
        self.assertEqual(len(evaluations), 3)
        self.assertEqual([item["true_label"] for item in evaluations], [1, 2, 3])
        self.assertTrue(
            all(
                0.0 <= item["true_label_confidence"] <= 1.0
                for item in evaluations
            )
        )

    def test_multiple_targets_are_distinct_and_proxy_correct(self):
        class EncodedPredictionModel(torch.nn.Module):
            def forward(self, inputs):
                predictions = inputs[:, 0].to(torch.long)
                logits = torch.zeros(len(inputs), 10)
                return logits.scatter_(1, predictions.unsqueeze(1), 1.0)

        dataset = TensorDataset(
            torch.tensor([[2.0], [3.0], [2.0], [2.0]]),
            torch.tensor([2, 2, 2, 2]),
        )
        targets = choose_cifar10_targets(
            EncodedPredictionModel(),
            dataset,
            target_label=2,
            target_count=3,
            target_index=None,
            target_indices=None,
            device=torch.device("cpu"),
        )
        self.assertEqual([target[0] for target in targets], [0, 2, 3])

    def test_target_indices_parser_requires_unique_integers(self):
        self.assertEqual(parse_target_indices("25, 42,108"), [25, 42, 108])
        with self.assertRaisesRegex(Exception, "unique"):
            parse_target_indices("25,25")

    def test_target_group_summary_reports_all_and_conditional_success_rates(self):
        before = [
            {"correct": True, "prediction": 2, "true_label_confidence": 0.8},
            {"correct": True, "prediction": 2, "true_label_confidence": 0.7},
            {"correct": False, "prediction": 4, "true_label_confidence": 0.1},
        ]
        after = [
            {"correct": False, "prediction": 4, "true_label_confidence": 0.2},
            {"correct": True, "prediction": 2, "true_label_confidence": 0.6},
            {"correct": True, "prediction": 2, "true_label_confidence": 0.5},
        ]
        summary = summarize_target_group(before, after)
        self.assertEqual(summary["success_count"], 1)
        self.assertAlmostEqual(summary["success_rate"], 1 / 3)
        self.assertAlmostEqual(summary["conditional_success_rate"], 1 / 2)
        self.assertAlmostEqual(summary["accuracy_before"], 2 / 3)
        self.assertAlmostEqual(summary["accuracy_after"], 2 / 3)

    def test_pood_training_label_can_follow_source_or_target(self):
        self.assertEqual(resolve_pood_training_label("source", 5, 2), 5)
        self.assertEqual(resolve_pood_training_label("target", 5, 2), 2)
        with self.assertRaises(ValueError):
            resolve_pood_training_label("unknown", 5, 2)

    def test_resnet18_exposes_512_dimensional_penultimate_features(self):
        model = build_model("cifar10").eval()
        features = extract_model_features(model, torch.randn(2, 3, 32, 32))
        self.assertEqual(features.shape, (2, 512))

    def test_feature_extraction_preserves_gradient_for_perturbation(self):
        model = build_model("cifar10").eval()
        inputs = torch.randn(1, 3, 32, 32, requires_grad=True)
        extract_model_features(model, inputs).sum().backward()
        self.assertIsNotNone(inputs.grad)

    def test_selection_uses_source_class_and_non_target_pseudo_label(self):
        target = torch.tensor([[1.0, 0.0]])
        features = torch.tensor(
            [
                [1.00, 0.00],
                [0.99, 0.01],
                [0.98, 0.02],
                [0.97, 0.03],
                [0.80, 0.20],
                [0.70, 0.30],
                [0.60, 0.40],
            ]
        )
        source_labels = torch.tensor([99, 12, 12, 12, 30, 30, 30])
        pseudo_labels = torch.tensor([3, 7, 7, 7, 2, 2, 2])
        source_class, poison_label, positions, similarities = select_cifar100_pood(
            target,
            features,
            source_labels,
            pseudo_labels,
            target_label=3,
            k=6,
            p=3,
        )
        self.assertEqual(source_class, 12)
        self.assertEqual(poison_label, 7)
        self.assertEqual(positions.tolist(), [1, 2, 3])
        self.assertEqual(len(similarities), 3)
        self.assertNotIn(0, positions.tolist())

    def test_cifar_pixel_round_trip_and_perturbation_budget(self):
        pixels = torch.rand(2, 3, 32, 32)
        mean = pixels.new_tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        std = pixels.new_tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
        normalized = (pixels - mean) / std
        self.assertTrue(
            torch.allclose(denormalize_cifar10(normalized), pixels, atol=1e-6)
        )
        delta = torch.full((1, 3, 32, 32), 4 / 255)
        perturbed = add_cifar10_pixel_perturbation(normalized, delta)
        pixel_delta = denormalize_cifar10(perturbed) - pixels
        self.assertLessEqual(float(pixel_delta.abs().max()), 4 / 255 + 1e-6)

    def test_proxy_loader_requires_cifar10_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(
                {
                    "dataset": "mnist",
                    "state_dict": build_model("mnist").state_dict(),
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "CIFAR-10 checkpoint"):
                load_cifar10_proxy(checkpoint, torch.device("cpu"))

    def test_latest_checkpoint_filters_out_other_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mnist_path = root / "newer" / "model.pt"
            cifar_path = root / "cifar" / "model.pt"
            mnist_path.parent.mkdir()
            cifar_path.parent.mkdir()
            torch.save({"dataset": "mnist"}, mnist_path)
            torch.save({"dataset": "cifar10"}, cifar_path)
            self.assertEqual(latest_cifar10_checkpoint(root), cifar_path)

    def test_integration_defaults_to_cifar_non_iid_setup(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.non_iid_alpha, DEFAULT_DIRICHLET_ALPHA)
        self.assertEqual(args.non_iid_alpha, 0.1)
        self.assertEqual(args.num_clients, 10)
        self.assertEqual(args.target_count, 1)
        self.assertIsNone(args.target_indices)
        self.assertGreaterEqual(args.k, args.p)
        self.assertAlmostEqual(args.perturb_epsilon, 8 / 255)
        self.assertEqual(args.retrieval_metric, "cosine")
        self.assertEqual(args.experiment_mode, "optimized")
        self.assertFalse(args.keep_unlearning_history)

    def test_control_and_l2_options_are_available(self):
        args = build_parser().parse_args(
            [
                "--retrieval-metric",
                "l2",
                "--experiment-mode",
                "unperturbed",
                "--keep-unlearning-history",
            ]
        )
        self.assertEqual(args.retrieval_metric, "l2")
        self.assertEqual(args.experiment_mode, "unperturbed")
        self.assertTrue(args.keep_unlearning_history)


if __name__ == "__main__":
    unittest.main()
