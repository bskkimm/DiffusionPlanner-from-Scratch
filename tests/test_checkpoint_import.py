import pytest
import torch
from test_model import config

from diffusion_planner import DiffusionPlanner
from diffusion_planner.checkpoint_import import import_official_state


def test_import_official_state_loads_matching_parameters():
    original = DiffusionPlanner(config())
    mapped = {}
    for key, value in original.state_dict().items():
        if key.startswith("encoder."):
            name = key.removeprefix("encoder.")
            for ours, theirs in [
                ("agents.", "neighbor_encoder."),
                ("static.", "static_encoder."),
                ("lanes.", "lane_encoder."),
                ("position.", "pos_emb."),
            ]:
                if name.startswith(ours):
                    name = theirs + name[len(ours) :]
                    break
            mapped["encoder.encoder." + name] = value.clone()
        else:
            name = key.removeprefix("decoder.").replace(
                "route_encoder.mixer.", "route_encoder.Mixer."
            )
            mapped["decoder.decoder.dit." + name] = value.clone()
    loaded = DiffusionPlanner(config())
    import_official_state(loaded, mapped)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    mapped["unexpected.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="unexpected"):
        import_official_state(loaded, mapped)
