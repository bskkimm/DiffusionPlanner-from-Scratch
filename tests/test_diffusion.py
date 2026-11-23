import torch

from diffusion_planner.diffusion import VPSchedule


def test_vp_schedule_boundaries_and_broadcast():
    schedule = VPSchedule()
    x = torch.ones(2, 3, 4, 4)
    times = torch.tensor([0.0, 1.0])
    mean, std = schedule.marginal_prob(x, times)
    torch.testing.assert_close(mean[0], x[0])
    assert std.shape == (2, 1, 1, 1)
    assert std[0].item() == 0
    assert 0.99 < std[1].item() < 1
    torch.testing.assert_close(mean[1, 0, 0, 0], torch.exp(torch.tensor(-5.025)))


def test_loss_handles_all_missing_neighbor_labels():
    from test_model import config, inputs

    from diffusion_planner import DiffusionPlanner

    cfg = config()
    model = DiffusionPlanner(cfg)
    losses = model.diffusion_loss(
        inputs(cfg),
        torch.randn(2, cfg.future_len, 4),
        torch.zeros(2, cfg.predicted_neighbor_num, cfg.future_len, 4),
        torch.ones(2, cfg.predicted_neighbor_num, cfg.future_len, dtype=torch.bool),
    )
    assert losses["neighbor_prediction_loss"].item() == 0
    losses["ego_planning_loss"].backward()
    assert torch.isfinite(losses["ego_planning_loss"])
