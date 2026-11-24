# Paper implementation brief

- Paper: Zheng et al., *Diffusion-Based Planning for Autonomous Driving with Flexible Guidance*, ICLR 2025, https://arxiv.org/abs/2501.15564.
- Official source: https://github.com/ZhengYinan-AIR/Diffusion-Planner, commit `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`, confirmed as remote main on 2026-11-24.
- Repository: `bskkimm/DiffusionPlanner-from-Scratch`, base `main`; public.
- Playbook: `bskkimm/paper-reimplementation-playbook`, commit `707c64b138d5e6c534af3d197fd4f278aab03d45`, confirmed as remote main on 2026-11-23.
- Audience: readers learning the paper and PyTorch implementation.
- Target: source-grounded implementation and teaching notebook. Operator and metric parity need separate evidence.
- Dataset: nuPlan v1.1; no dataset, split or simulator availability established. No dataset download authorized.
- Weights: random initialization; no pretrained weights loaded or verified.
- Hardware: available PyTorch 2.9.1+cu130 reports CUDA unavailable. GPU runner not established.
- Budget: no full-training or paid-compute authorization; synthetic CPU verification only by default.
- Training recipe: paper Appendix C.1 reports 1 million nuPlan scenarios, 8 A100 GPUs, global batch 2048, 500 epochs, 5-epoch warmup and AdamW at 5e-4; local practical runner differs. Official simulator evaluation remains unverified.
- Evaluation: real-data and closed-loop official metrics remain unmeasured.
- Notebook: local architecture definitions, fresh-process CPU execution, package loss/gradient parity, optimization and an inference walkthrough are tested. Final illustrations and visual checks are deferred per skill default.
- Reviewer: existing pra cloud configuration uses Claude for own repositories, skips drafts and posts COMMENT reviews. Current-head review is pending. GitHub reports no main-branch protection or rulesets.
- Delivery authorization: user-invoked skill authorizes merge of this paper project's PRs after all required gates pass, overriding the playbook's generic no-merge instruction.
