import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from fl_sim.PUA import (
    DEFAULT_DIRICHLET_ALPHA,
    PoisonedClientDataset,
    build_parser,
    compare_per_label,
    latest_recursive_checkpoint,
    validate_args,
)


class PuaIntegrationUnitTests(unittest.TestCase):
    def test_integration_defaults_to_requested_non_iid_partition_strength(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.non_iid_alpha, DEFAULT_DIRICHLET_ALPHA)
        self.assertEqual(args.non_iid_alpha, 0.1)
        self.assertGreaterEqual(args.k, args.p)

    def test_poisoned_dataset_tracks_every_injected_local_index(self):
        benign = TensorDataset(torch.arange(3), torch.tensor([0, 1, 2]))
        poison_images = torch.tensor([10, 11])
        poison_labels = torch.tensor([8, 9])
        dataset = PoisonedClientDataset(
            benign, poison_images, poison_labels, repeats=3
        )
        self.assertEqual(len(dataset), 9)
        self.assertEqual(dataset.poison_local_indices, list(range(3, 9)))
        self.assertEqual([dataset[index][1] for index in range(3, 9)], [8, 9] * 3)

    def test_per_label_comparison_reports_accuracy_drop(self):
        before = {"0": {"sample_count": 4, "accuracy": 0.75}}
        after = {"0": {"sample_count": 4, "accuracy": 0.25}}
        self.assertEqual(
            compare_per_label(before, after)["0"],
            {
                "sample_count": 4,
                "before_accuracy": 0.75,
                "after_accuracy": 0.25,
                "accuracy_drop": 0.5,
            },
        )

    def test_latest_checkpoint_searches_nested_run_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old" / "model.pt"
            new = Path(directory) / "nested" / "new" / "model.pt"
            old.parent.mkdir(parents=True)
            new.parent.mkdir(parents=True)
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            self.assertEqual(latest_recursive_checkpoint(directory), new)

    def test_validation_rejects_invalid_pood_shape_constraints(self):
        args = Namespace(
            target_label=3,
            malicious_client_id=0,
            num_clients=10,
            non_iid_alpha=0.1,
            candidate_count=10,
            feature_batch_size=8,
            p=6,
            k=5,
            perturb_steps=1,
            perturb_lr=0.01,
            perturb_epsilon=0.3,
            poison_repeats=1,
            rounds=1,
            local_epochs=1,
            learning_rate=0.01,
            batch_size=8,
            eval_batch_size=8,
            unlearning_delta_t=1,
            calibration_local_epochs=1,
        )
        with self.assertRaisesRegex(ValueError, "p <= k"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
