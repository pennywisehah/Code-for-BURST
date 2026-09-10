import unittest

import torch
from PIL import Image
from torch.utils.data import Dataset

from fl_sim.STL_PUA import build_stl_parser
from fl_sim.STL_PUA_TargetLabel import build_target_label_parser
from fl_sim.stl_pood import (
    STL10_TO_CIFAR10,
    build_stl10_cifar_transform,
    collect_stl10_candidates,
    select_stl10_pood,
)


class SyntheticStl10(Dataset):
    def __init__(self):
        self.samples = [
            (torch.full((3, 32, 32), float(label)), label)
            for label in range(10)
            for _ in range(3)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class StlPoodTests(unittest.TestCase):
    def test_mapping_covers_nine_classes_and_excludes_monkey(self):
        self.assertEqual(
            STL10_TO_CIFAR10,
            {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 8: 8, 9: 9},
        )
        self.assertNotIn(7, STL10_TO_CIFAR10)

    def test_transform_resizes_and_normalizes_for_cifar10(self):
        image = Image.new("RGB", (96, 96), color=(128, 64, 32))
        transformed = build_stl10_cifar_transform()(image)
        self.assertEqual(transformed.shape, (3, 32, 32))
        self.assertTrue(torch.isfinite(transformed).all())

    def test_collection_excludes_monkey_and_target_mapped_class(self):
        _, source_labels, mapped_labels, source_indices = collect_stl10_candidates(
            SyntheticStl10(),
            target_label=2,
            candidate_count=20,
            seed=42,
        )
        self.assertEqual(len(source_indices), 20)
        self.assertNotIn(7, source_labels.tolist())
        self.assertNotIn(1, source_labels.tolist())
        self.assertNotIn(2, mapped_labels.tolist())

    def test_selection_uses_mapped_true_label_not_proxy_prediction(self):
        target = torch.tensor([[1.0, 0.0]])
        features = torch.tensor(
            [
                [1.00, 0.00],
                [0.99, 0.01],
                [0.98, 0.02],
                [0.80, 0.20],
                [0.70, 0.30],
            ]
        )
        source_labels = torch.tensor([5, 5, 5, 8, 8])
        mapped_labels = torch.tensor([5, 5, 5, 8, 8])
        source_label, poison_label, positions, similarities = select_stl10_pood(
            target,
            features,
            source_labels,
            mapped_labels,
            k=5,
            p=3,
        )
        self.assertEqual(source_label, 5)
        self.assertEqual(poison_label, 5)
        self.assertEqual(positions.tolist(), [0, 1, 2])
        self.assertEqual(len(similarities), 3)

    def test_stl_entrypoint_sets_safe_defaults(self):
        args = build_stl_parser().parse_args([])
        self.assertEqual(args.pood_dataset, "stl10")
        self.assertEqual(args.stl10_split, "test")
        self.assertEqual(args.output_dir, "stl_pua_runs")
        self.assertEqual(args.non_iid_alpha, 0.1)
        self.assertEqual(args.pood_training_label, "source")

    def test_target_label_entrypoint_changes_label_treatment_defaults(self):
        args = build_target_label_parser().parse_args([])
        self.assertEqual(args.pood_dataset, "stl10")
        self.assertEqual(args.pood_training_label, "target")
        self.assertEqual(args.output_dir, "stl_pua_target_label_runs")
        self.assertEqual(args.experiment_mode, "optimized")
        self.assertEqual(args.non_iid_alpha, 0.1)


if __name__ == "__main__":
    unittest.main()
