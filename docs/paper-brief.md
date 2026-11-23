# Paper implementation brief

- Paper: Zheng et al., *Diffusion-Based Planning for Autonomous Driving with Flexible Guidance*, ICLR 2025, https://arxiv.org/abs/2501.15564.
- Official source: https://github.com/ZhengYinan-AIR/Diffusion-Planner, cached commit `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`; remote freshness unverified.
- Repository: `bskkimm/DiffusionPlanner-from-Scratch`, base `main`; existing remote visibility not verified.
- Playbook: `bskkimm/paper-reimplementation-playbook`, cached commit `707c64b138d5e6c534af3d197fd4f278aab03d45`; remote freshness unverified.
- Audience: readers learning the paper and PyTorch implementation.
- Target: source-grounded implementation and teaching notebook. Operator and metric parity need separate evidence.
- Dataset: nuPlan v1.1; no dataset, split or simulator availability established. No dataset download authorized.
- Weights: random initialization; no pretrained weights loaded or verified.
- Hardware: available PyTorch 2.9.1+cu130 reports CUDA unavailable. GPU runner not established.
- Budget: no full-training or paid-compute authorization; synthetic CPU verification only by default.
- Training recipe and official evaluation: require mapping and verification against the pinned upstream configuration and simulator.
- Evaluation: real-data and closed-loop official metrics remain unmeasured.
- Notebook: local architecture definitions exist; execution, parity, full optimization and inference workflows require verification. Final illustrations and visual checks are deferred per skill default.
- Reviewer and branch protection: not yet verified; network unavailable during initial audit.
- Delivery authorization: user-invoked skill authorizes merge of this paper project's PRs after all required gates pass, overriding the playbook's generic no-merge instruction.
