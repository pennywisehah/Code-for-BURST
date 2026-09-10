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
    def test_delta_t_accumulates_interval_updates_and_final_partial_window(self):
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
        self.assertEqual(
            [snapshot["start_round"] for snapshot in payload["snapshots"]],
            [1, 3, 5],
        )
        self.assertEqual(
            [snapshot["interval_round_count"] for snapshot in payload["snapshots"]],
            [2, 2, 1],
        )
        self.assertEqual(
            [
                snapshot["updates"][0]["weight"].item()
                for snapshot in payload["snapshots"]
            ],
            [3.0, 7.0, 5.0],
        )
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(
            payload["history_update_semantics"], "per_client_interval_sum"
        )

    def test_interval_accumulation_tracks_clients_by_id_across_order_changes(self):
        recorder = FedEraserHistoryRecorder(
            {"weight": torch.tensor([0.0])}, delta_t=2
        )
        recorder.record_round(
            1,
            client_ids=[0, 1],
            updates=[
                {"weight": torch.tensor([1.0])},
                {"weight": torch.tensor([10.0])},
            ],
            sample_counts=[4, 8],
        )
        recorder.record_round(
            2,
            client_ids=[1, 0],
            updates=[
                {"weight": torch.tensor([20.0])},
                {"weight": torch.tensor([2.0])},
            ],
            sample_counts=[8, 4],
        )
        snapshot = recorder.build_payload(
            {"aggregation": "fedavg"}, {"weight": torch.tensor([0.0])}
        )["snapshots"][0]
        updates_by_client = dict(zip(snapshot["client_ids"], snapshot["updates"]))
        self.assertEqual(updates_by_client[0]["weight"].item(), 3.0)
        self.assertEqual(updates_by_client[1]["weight"].item(), 30.0)

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

    def test_interval_history_runs_nonzero_calibration_without_nan(self):
        initial_state = build_model("mnist").state_dict()
        round_update = {
            name: torch.full_like(value, 1e-4)
            for name, value in initial_state.items()
            if value.is_floating_point()
        }
        recorder = FedEraserHistoryRecorder(initial_state, delta_t=2)
        for round_number in (1, 2):
            recorder.record_round(
                round_number,
                client_ids=[0],
                updates=[round_update],
                sample_counts=[4],
            )
        history = recorder.build_payload(
            {"aggregation": "fedavg", "dataset": "mnist"}, initial_state
        )
        clients = [
            TensorDataset(
                torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 0, 1])
            )
        ]
        reconstructed, calibration_history = federaser_unlearn(
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
        self.assertTrue(
            all(
                torch.isfinite(value).all()
                for value in reconstructed.values()
                if value.is_floating_point()
            )
        )
        self.assertEqual(calibration_history[0]["history_interval_start"], 1)
        self.assertEqual(calibration_history[0]["history_interval_end"], 2)
        self.assertGreater(
            calibration_history[0]["historical_layer_norm_sum"], 0.0
        )

if __name__ == "__main__":
    unittest.main()
