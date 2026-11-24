"""Explicit parameter-name adapter for the pinned official code snapshot."""


def import_official_state(model, state):
    """Load an official model state dict after strict name and shape validation.

    Accepts the full model dictionary, optionally with a DDP ``module.`` prefix.
    This maps architecture names only; actual released weights and end-to-end
    parity have not been verified. Checkpoint files must come from trusted sources.
    """
    result = {}
    prefixes = {
        "encoder.encoder.neighbor_encoder.": "encoder.agents.",
        "encoder.encoder.static_encoder.": "encoder.static.",
        "encoder.encoder.lane_encoder.": "encoder.lanes.",
        "encoder.encoder.pos_emb.": "encoder.position.",
        "encoder.encoder.": "encoder.",
        "decoder.decoder.dit.": "decoder.",
    }
    for key, value in state.items():
        name = key.removeprefix("module.")
        for source, target in prefixes.items():
            if name.startswith(source):
                name = target + name[len(source) :]
                break
        else:
            raise ValueError(f"unexpected official parameter: {key}")
        name = name.replace("decoder.route_encoder.Mixer.", "decoder.route_encoder.mixer.")
        if name in result:
            raise ValueError(f"duplicate mapped parameter: {name}")
        result[name] = value
    expected = model.state_dict()
    missing = expected.keys() - result.keys()
    extra = result.keys() - expected.keys()
    mismatched = [
        name
        for name in expected.keys() & result.keys()
        if expected[name].shape != result[name].shape
    ]
    if missing or extra or mismatched:
        raise ValueError(
            f"checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(extra)}, shapes={sorted(mismatched)}"
        )
    model.load_state_dict(result, strict=True)
