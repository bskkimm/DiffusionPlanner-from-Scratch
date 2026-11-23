import torch

from diffusion_planner import DiffusionPlanner, PlannerConfig
from diffusion_planner.model.dit import RouteEncoder
from diffusion_planner.model.encoder import ContextEncoder


def config() -> PlannerConfig:
    return PlannerConfig(
        future_len=3,
        time_len=4,
        agent_num=3,
        static_objects_num=2,
        lane_len=4,
        lane_num=3,
        route_num=2,
        predicted_neighbor_num=2,
        hidden_dim=32,
        num_heads=4,
        encoder_depth=1,
        decoder_depth=1,
    )


def inputs(model_config: PlannerConfig, batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "ego_current_state": torch.randn(batch, 10),
        "neighbor_agents_past": torch.randn(
            batch, model_config.agent_num, model_config.time_len, 11
        ),
        "static_objects": torch.randn(batch, model_config.static_objects_num, 10),
        "lanes": torch.randn(batch, model_config.lane_num, model_config.lane_len, 12),
        "lanes_speed_limit": torch.rand(batch, model_config.lane_num, 1),
        "lanes_has_speed_limit": torch.ones(batch, model_config.lane_num, 1, dtype=torch.bool),
        "route_lanes": torch.randn(batch, model_config.route_num, model_config.lane_len, 12),
    }


def test_context_and_route_encoders_match_official_token_shapes() -> None:
    model_config = config()
    batch = inputs(model_config)
    context = ContextEncoder(model_config)(batch)
    assert context["encoding"].shape == (
        2,
        model_config.agent_num + model_config.static_objects_num + model_config.lane_num,
        model_config.hidden_dim,
    )
    assert context["mask"].shape == (
        2,
        model_config.agent_num + model_config.static_objects_num + model_config.lane_num,
    )
    assert RouteEncoder(model_config)(batch["route_lanes"]).shape == (2, model_config.hidden_dim)


def test_official_training_contract_and_loss() -> None:
    model_config = config()
    model = DiffusionPlanner(model_config).train()
    batch = inputs(model_config)
    ego_future = torch.randn(2, model_config.future_len, 4)
    neighbor_future = torch.randn(
        2, model_config.predicted_neighbor_num, model_config.future_len, 4
    )
    neighbor_mask = torch.zeros(
        2, model_config.predicted_neighbor_num, model_config.future_len, dtype=torch.bool
    )
    loss = model.diffusion_loss(batch, ego_future, neighbor_future, neighbor_mask)
    total = loss["ego_planning_loss"] + loss["neighbor_prediction_loss"]
    total.backward()
    assert total.ndim == 0
    assert torch.isfinite(total)


def test_denoiser_preserves_official_trajectory_shape() -> None:
    model_config = config()
    model = DiffusionPlanner(model_config).eval()
    batch = inputs(model_config)
    encoded, output = model(batch)
    noisy = torch.randn(2, model_config.participant_num, model_config.trajectory_dim)
    denoised = model.denoise(noisy, torch.full((2,), 0.5), encoded, batch)
    assert output["current_states"].shape == (2, model_config.participant_num, 4)
    assert denoised.shape == (2, model_config.participant_num, model_config.trajectory_dim)


def test_mixer_dropout_one_leaves_residual_finite_and_unchanged():
    from diffusion_planner.model.mixer import MixerBlock

    mixer = MixerBlock(4, 8, 1.0).train()
    inputs = torch.randn(2, 4, 8)
    output = mixer(inputs)
    torch.testing.assert_close(output, inputs)


def test_mixer_training_matches_pinned_upstream_fixture():
    """CPU fixture: official a3a621f, timm 1.0.10, torch 2.9.1."""
    from diffusion_planner.model.mixer import MixerBlock

    torch.manual_seed(42)
    mixer = MixerBlock(4, 8, 0.1).train()
    sample = torch.randn(2, 4, 8)
    torch.manual_seed(99)
    actual = mixer(sample)[0, 0]
    expected = torch.tensor(
        [
            0.51846385,
            1.28768873,
            -1.56435347,
            -0.46910012,
            -0.60699219,
            -1.33802152,
            -0.24616662,
            -0.19193977,
        ]
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_drop_path_matches_timm_seeded_bernoulli_mask():
    from diffusion_planner.model.mixer import DropPath

    sample = torch.ones(16, 3, 2)
    torch.manual_seed(17)
    expected = torch.empty(16, 1, 1).bernoulli_(0.75).div_(0.75) * sample
    torch.manual_seed(17)
    actual = DropPath(0.25).train()(sample)
    torch.testing.assert_close(actual, expected)


def test_context_encoder_training_matches_pinned_upstream_fixture():
    """Reduced CPU encoder fixture from upstream a3a621f, timm 1.0.10."""
    cfg = config()
    model = ContextEncoder(cfg).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = torch.linspace(-0.1, 0.1, parameter.numel()).reshape(parameter.shape)
            parameter.copy_(1 + value if "norm" in name and name.endswith("weight") else value)
    torch.manual_seed(111)
    batch = {
        "neighbor_agents_past": torch.randn(2, 3, 4, 11),
        "static_objects": torch.randn(2, 2, 10),
        "lanes": torch.randn(2, 3, 4, 12),
        "lanes_speed_limit": torch.rand(2, 3, 1),
        "lanes_has_speed_limit": torch.ones(2, 3, 1, dtype=torch.bool),
    }
    batch["neighbor_agents_past"][0, 0] = 0
    batch["lanes"][1, 1] = 0
    torch.manual_seed(222)
    output = model(batch)["encoding"][0, 0, :8]
    expected = torch.tensor(
        [
            -1.66959822,
            -0.99707896,
            -1.47608531,
            -1.37717569,
            -1.27683043,
            -1.17505026,
            -1.07183409,
            -0.96718276,
        ]
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_dit_block_matches_pinned_upstream_fixture():
    """CPU float32 fixture from the released DiT block and timm 1.0.10."""
    from diffusion_planner.model.dit import DiTBlock

    model = DiTBlock(dim=32, heads=4, dropout=0).eval()
    assert model.mlp1.act.approximate == "tanh"
    assert model.mlp2.act.approximate == "tanh"
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = torch.linspace(-0.1, 0.1, parameter.numel()).reshape(parameter.shape)
            parameter.copy_(1 + value if "norm" in name and name.endswith("weight") else value)
    torch.manual_seed(111)
    x = torch.randn(2, 3, 32)
    context = torch.randn(2, 4, 32)
    condition = torch.randn(2, 32)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    actual = model(x, context, condition, mask)[0, 0, :8]
    expected = torch.tensor(
        [
            -0.0359268337,
            -0.0339872427,
            -0.0320475921,
            -0.0301079862,
            -0.0281683393,
            -0.0262286924,
            -0.0242890641,
            -0.0223494470,
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_full_dit_training_matches_pinned_upstream_fixture():
    """Reduced full decoder CPU fixture, official a3a621f and timm 1.0.10."""
    from diffusion_planner.model.dit import TrajectoryDiT

    model = TrajectoryDiT(config()).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = torch.linspace(-0.1, 0.1, parameter.numel()).reshape(parameter.shape)
            parameter.copy_(1 + value if "norm" in name and name.endswith("weight") else value)
    torch.manual_seed(111)
    noisy = torch.randn(2, 3, 16)
    time = torch.tensor([0.2, 0.7])
    context = torch.randn(2, 8, 32)
    route = torch.randn(2, 2, 4, 12)
    mask = torch.tensor([[False, True], [False, False]])
    torch.manual_seed(222)
    output = model(noisy, time, context, route, mask)[0, 0, :8]
    expected = torch.tensor(
        [
            -0.76387769,
            -0.65807623,
            -0.55227476,
            -0.44647336,
            -0.34067196,
            -0.23487051,
            -0.12906905,
            -0.02326762,
        ]
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
