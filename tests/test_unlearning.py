import unittest

import torch
from torch.utils.data import TensorDataset

from fl_sim.unlearning import (
    FedEraserHistoryRecorder,
    calibrate_update_direction,
    remove_local_samples,
)


class FedEraserTests(unittest.TestCase):
    def test_delta_t_stores_interval_rounds_and_final_round(self):
        initial = {"weight": torch.tensor([0.0])}
        recorder = FedEraserHistoryRecorder(initial, delta_t=2)
        for round_number in range(1, 6):
            recorder.record_round(
                round_number,
                client_ids=[0],
                updates=[{"weight": torch.tensor([float(round_number)])}],
                sample_counts=[4],
            )
        payload = recorder.build_payload(
            {"aggregation": "fedavg"}, {"weight": torch.tensor([1.0])}
        )
        self.assertEqual(
            [snapshot["round"] for snapshot in payload["snapshots"]], [2, 4, 5]
        )

    def test_calibration_uses_old_norm_and_new_direction(self):
        old = {"weight": torch.tensor([3.0, 4.0])}
        new = {"weight": torch.tensor([0.0, 2.0])}
        calibrated = calibrate_update_direction(old, new)
        self.assertTrue(torch.allclose(calibrated["weight"], torch.tensor([0.0, 5.0])))

    def test_remove_local_samples_keeps_only_retained_positions(self):
        dataset = TensorDataset(torch.arange(5), torch.arange(5))
        retained = remove_local_samples(dataset, [1, 3])
        self.assertEqual(len(retained), 3)
        self.assertEqual([int(retained[index][0]) for index in range(3)], [0, 2, 4])

    def test_partial_removal_rejects_all_samples(self):
        dataset = TensorDataset(torch.arange(2), torch.arange(2))
        with self.assertRaises(ValueError):
            remove_local_samples(dataset, [0, 1])

if __name__ == "__main__":
    unittest.main()
