import unittest
from collections import OrderedDict

import torch

from fl_sim.aggregation import fedavg, krum, trimmed_mean
from fl_sim.config import ExperimentConfig


def update(values):
    return OrderedDict(weight=torch.tensor(values, dtype=torch.float32))


class AggregationTests(unittest.TestCase):
    def test_fedavg_uses_sample_weights(self):
        result = fedavg([update([0.0, 2.0]), update([10.0, 4.0])], [3, 1])
        torch.testing.assert_close(result["weight"], torch.tensor([2.5, 2.5]))

    def test_trimmed_mean_removes_extremes(self):
        updates = [update([value]) for value in [1.0, 1.1, 0.9, 1.2, 100.0]]
        result = trimmed_mean(updates, 0.2, chunk_size=2)
        self.assertAlmostEqual(result["weight"].item(), 1.1)

    def test_krum_ignores_outlier(self):
        updates = [
            update([1.0, 1.0]),
            update([1.1, 1.0]),
            update([0.9, 1.0]),
            update([1.0, 0.9]),
            update([50.0, 50.0]),
        ]
        selected = krum(updates, byzantine=1)["weight"]
        self.assertLess(abs(selected[0].item() - 1.0), 0.2)
        self.assertLess(abs(selected[1].item() - 1.0), 0.2)

    def test_krum_rejects_unsafe_client_count(self):
        with self.assertRaisesRegex(ValueError, "n >= 2f"):
            krum([update([0.0]), update([0.1]), update([100.0])], byzantine=1)

    def test_config_rejects_invalid_dataset(self):
        config = ExperimentConfig(dataset="unknown")
        with self.assertRaises(ValueError):
            config.validate()

    def test_config_requires_one_client_per_label(self):
        config = ExperimentConfig(
            dataset="mnist", partition="label_per_client", num_clients=9
        )
        with self.assertRaisesRegex(ValueError, "num_clients=10"):
            config.validate()

if __name__ == "__main__":
    unittest.main()
