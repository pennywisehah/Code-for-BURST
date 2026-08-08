import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from fl_sim.data import (
    _dirichlet_partition,
    _iid_partition,
    _label_per_client_partition,
    unique_dataset_labels,
)
from fl_sim.model import (
    CNN2,
    apply_update,
    build_model,
    evaluate_per_class,
    train_local,
)


class IdentityLogitModel(nn.Module):
    def forward(self, inputs):
        return inputs


class ModelTests(unittest.TestCase):
    def test_cnn2_mnist_shape(self):
        model = build_model("mnist")
        self.assertIsInstance(model, CNN2)
        self.assertEqual(model(torch.randn(2, 1, 28, 28)).shape, (2, 10))

    def test_cnn2_fashionmnist_shape(self):
        model = build_model("fashionmnist")
        self.assertEqual(model(torch.randn(2, 1, 28, 28)).shape, (2, 10))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_cnn2_mnist_shape_on_mps(self):
        model = build_model("mnist").to("mps")
        inputs = torch.randn(2, 1, 28, 28, device="mps")
        self.assertEqual(model(inputs).shape, (2, 10))

    def test_resnet18_cifar10_shape(self):
        model = build_model("cifar10")
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))

    def test_resnet18_cifar100_shape(self):
        model = build_model("cifar100")
        self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 100))

    def test_cnn2_local_training_returns_applicable_update(self):
        model = build_model("mnist")
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        dataset = TensorDataset(
            torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
        )
        update = train_local(
            global_state=state,
            dataset=dataset,
            model_name="mnist",
            local_epochs=1,
            learning_rate=0.01,
            batch_size=4,
            momentum=0.0,
            weight_decay=0.0,
            num_workers=0,
            device=torch.device("cpu"),
            seed=1,
        )
        updated_state = apply_update(state, update)
        self.assertEqual(set(updated_state), set(state))
        self.assertTrue(any(torch.count_nonzero(value) for value in update.values()))

    def test_evaluate_per_class_reports_accuracy_and_empty_classes(self):
        logits = torch.tensor(
            [[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 3.0, 1.0]]
        )
        targets = torch.tensor([0, 0, 1])
        dataset = TensorDataset(logits, targets)
        with patch("fl_sim.model.build_model", return_value=IdentityLogitModel()):
            metrics = evaluate_per_class(
                state={},
                dataset=dataset,
                model_name="mnist",
                num_classes=3,
                batch_size=2,
                num_workers=0,
                device=torch.device("cpu"),
            )
        self.assertEqual(metrics["0"], {"sample_count": 2, "accuracy": 0.5})
        self.assertEqual(metrics["1"], {"sample_count": 1, "accuracy": 1.0})
        self.assertEqual(metrics["2"], {"sample_count": 0, "accuracy": None})


class PartitionTests(unittest.TestCase):
    def test_iid_partition_uses_every_sample_once(self):
        parts = _iid_partition(20, 4, torch.Generator().manual_seed(1))
        self.assertEqual(sorted(index for part in parts for index in part), list(range(20)))
        self.assertEqual([len(part) for part in parts], [5, 5, 5, 5])

    def test_dirichlet_partition_uses_every_sample_once(self):
        targets = torch.tensor([label for label in range(4) for _ in range(20)])
        parts = _dirichlet_partition(
            targets,
            num_clients=4,
            num_classes=4,
            alpha=0.5,
            generator=torch.Generator().manual_seed(1),
        )
        self.assertEqual(
            sorted(index for part in parts for index in part), list(range(len(targets)))
        )
        self.assertTrue(all(parts))

    def test_label_per_client_assigns_exactly_one_class(self):
        targets = torch.tensor([label for label in range(4) for _ in range(5)])
        parts = _label_per_client_partition(
            targets,
            num_clients=4,
            num_classes=4,
            generator=torch.Generator().manual_seed(1),
        )
        self.assertEqual(
            sorted(index for part in parts for index in part), list(range(len(targets)))
        )
        for client_id, indices in enumerate(parts):
            self.assertEqual(set(targets[indices].tolist()), {client_id})

    def test_label_per_client_rejects_mismatched_client_count(self):
        with self.assertRaisesRegex(ValueError, "num_clients"):
            _label_per_client_partition(
                torch.tensor([0, 0, 1, 1]),
                num_clients=3,
                num_classes=2,
                generator=torch.Generator().manual_seed(1),
            )

    def test_unique_dataset_labels_supports_subsets(self):
        dataset = TensorDataset(torch.arange(6), torch.tensor([0, 1, 1, 2, 2, 2]))
        subset = torch.utils.data.Subset(dataset, [1, 2])
        self.assertEqual(unique_dataset_labels(subset), [1])


if __name__ == "__main__":
    unittest.main()
