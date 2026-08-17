import unittest

import torch
from torch.utils.data import TensorDataset

from fl_sim.model import build_model
from fl_sim.unlearning import (
    FedEraserHistoryRecorder,
    calibrate_update_direction,
    federaser_unlearn,
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

    def test_reconstruction_without_forgetting_accepts_an_empty_request(self):
        initial_state = build_model("mnist").state_dict()
        zero_update = {
            name: torch.zeros_like(value)
            for name, value in initial_state.items()
            if value.is_floating_point()
        }
        history = {
            "method": "federaser",
            "config": {"aggregation": "fedavg", "dataset": "mnist"},
            "initial_state": initial_state,
            "snapshots": [
                {
                    "round": 1,
                    "client_ids": [0],
                    "sample_counts": [2],
                    "updates": [zero_update],
                }
            ],
        }
        clients = [
            TensorDataset(torch.randn(2, 1, 28, 28), torch.tensor([0, 1]))
        ]
        _, calibration_history = federaser_unlearn(
            history=history,
            client_datasets=clients,
            forget_client_id=0,
            forget_local_indices=None,
            forget_all_client_data=False,
            calibration_local_epochs=1,
            calibration_learning_rate=0.01,
            batch_size=2,
            momentum=0.0,
            weight_decay=0.0,
            num_workers=0,
            device=torch.device("cpu"),
            seed=42,
            reconstruct_without_forgetting=True,
        )
        self.assertEqual(calibration_history[0]["retained_samples"], 2)

if __name__ == "__main__":
    unittest.main()
