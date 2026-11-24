# Deferred notebook figure inventory

Final figures and visual checks are deferred until explicitly requested. Full
shapes refer to `PlannerConfig` defaults; examples are illustrative, not measured.
Each figure should have editable source, an offline attachment and a GitHub-viewable
preview. Verify numerical examples, arrows and legibility when producing them.

| ID / placement | Learner question and worked example | Source and full shape |
| --- | --- | --- |
| pipeline / introduction | How do observations, route, noised futures and loss flow through training and sampling? Show both branches and current-state constraint. | `model/planner.py`; 11 participants, 81 states, 4 channels. |
| inputs / before encoder | How are padded histories and future heading encoded? Trace `(x,y,yaw)` to `(x,y,cos,sin)` and mask zero labels first. | `data.py`, `model/encoder.py`; 32 histories × 21 × 11. |
| mixer / before MixerBlock | How do token mixing and channel mixing transpose and connect residual paths? Use a labeled small matrix. | `model/mixer.py`; lane and agent inputs use 64 tokens and 128 channels internally. |
| context / before ContextEncoder | How do agent, static and lane masks form context for attention? Include an all-padded example. | `model/encoder.py`; 32 + 5 + 70 tokens. |
| route-time / before RouteEncoder | How do route features combine with diffusion-time conditioning? Trace separate route and time paths. | `model/dit.py`; 25 route lanes × 20 points. |
| dit / before DiTBlock | How do adaptive normalization, residual gates, self- and cross-attention connect? Show identity at zero initialization. | `model/dit.py`; hidden size 192, 6 heads. |
| loss / before diffusion_loss | How do forward noise, x-start predictions, masks and ego/neighbor reductions form the objective? Calculate one toy reduction. | `model/planner.py`, `diffusion.py`; 80 future states. |
| sample / final section | How are reverse-time states updated and current state held fixed? Distinguish practical sampler from upstream DPM-Solver++. | `sampling.py`, upstream `model/diffusion_utils/sampling.py`. |
| recovery / final section | What is saved at an epoch boundary, and what does a resumed run restore? Show data identity check and random states. | `training.py`, `train.py`; CPU zero-worker tested scope. |
