import torch

from diffusion_planner.data import encode_future


def test_encode_future_preserves_padding_before_heading_conversion():
    future = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, torch.pi / 2]]])
    encoded, missing = encode_future(future, mask_padding=True)
    torch.testing.assert_close(encoded[0, 0], torch.zeros(4))
    torch.testing.assert_close(encoded[0, 1], torch.tensor([1.0, 2.0, 0.0, 1.0]), atol=1e-6, rtol=0)
    assert missing.tolist() == [[True, False]]


def test_ego_origin_is_not_treated_as_missing():
    encoded, missing = encode_future(torch.zeros(1, 1, 3), mask_padding=False)
    torch.testing.assert_close(encoded, torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]))
    assert not missing.any()


def test_npz_dataset_converts_targets_and_preserves_mask(tmp_path):
    import json

    import numpy as np

    from diffusion_planner.data import NuPlanDataset

    np.savez(
        tmp_path / "sample.npz",
        ego_current_state=np.ones(10, dtype="float32"),
        ego_agent_future=np.array([[0, 0, 0]], dtype="float32"),
        neighbor_agents_past=np.ones((2, 2, 11), dtype="float32"),
        neighbor_agents_future=np.array([[[0, 0, 0]], [[1, 2, 0]]], dtype="float32"),
        static_objects=np.ones((1, 10), dtype="float32"),
        lanes=np.ones((1, 2, 12), dtype="float32"),
        lanes_speed_limit=np.ones((1, 1), dtype="float32"),
        lanes_has_speed_limit=np.ones((1, 1), dtype="bool"),
        route_lanes=np.ones((1, 2, 12), dtype="float32"),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(["sample.npz"]))
    dataset = NuPlanDataset(tmp_path, manifest, past_neighbors=2, predicted_neighbors=2)
    inputs, ego, neighbors, mask = dataset[0]
    assert inputs["neighbor_agents_past"].shape == (2, 2, 11)
    torch.testing.assert_close(ego[0], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert mask.tolist() == [[True], [False]]
    assert neighbors[0].count_nonzero() == 0


def test_official_observation_normalization_preserves_padding():
    from diffusion_planner.data import normalize_observations

    original = {
        "ego_current_state": torch.tensor([[10.0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),
        "static_objects": torch.zeros(1, 1, 10),
    }
    normalized = normalize_observations(original)
    assert normalized["ego_current_state"][0, 0] == 0
    assert normalized["ego_current_state"][0, 2] == 1
    assert normalized["static_objects"].count_nonzero() == 0
    assert original["ego_current_state"][0, 0] == 10


def test_dataset_can_expose_raw_yaw_for_augmentation(tmp_path):
    import json

    import numpy as np

    from diffusion_planner.data import NuPlanDataset

    np.savez(
        tmp_path / "a.npz",
        ego_current_state=np.ones(10, dtype="float32"),
        ego_agent_future=np.ones((21, 3), dtype="float32"),
        neighbor_agents_past=np.ones((1, 2, 11), dtype="float32"),
        neighbor_agents_future=np.ones((1, 21, 3), dtype="float32"),
        static_objects=np.ones((1, 10), dtype="float32"),
        lanes=np.ones((1, 2, 12), dtype="float32"),
        lanes_speed_limit=np.ones((1, 1), dtype="float32"),
        lanes_has_speed_limit=np.ones((1, 1), dtype="bool"),
        route_lanes=np.ones((1, 2, 12), dtype="float32"),
    )
    manifest = tmp_path / "list.json"
    manifest.write_text(json.dumps(["a.npz"]))
    data = NuPlanDataset(
        tmp_path, manifest, past_neighbors=1, predicted_neighbors=1, raw_futures=True
    )
    assert data[0][1].shape[-1] == 3
    assert data[0][2].shape[-1] == 3
