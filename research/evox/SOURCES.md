# Implementation provenance

Paper: **EvoX: Meta-Evolution for Automated Discovery**, Shu Liu, Shubham
Agarwal et al.; Algorithm 1, Sections 3–4, and Appendix C supplied with the request.

Official implementation: [skydiscover-ai/skydiscover](https://github.com/skydiscover-ai/skydiscover),
inspected at commit [`0d932b690670a7e544388ad362e9876c8bd256a0`](https://github.com/skydiscover-ai/skydiscover/tree/0d932b690670a7e544388ad362e9876c8bd256a0).
The repository was cloned and its implementation read; no benchmark package is
required at runtime here.

| Official source at that commit | Use here |
| --- | --- |
| [initial_search_strategy.py](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/evox/database/initial_search_strategy.py) | Adapted scalar uniform parent/inspiration sampling in `initial_strategy.py`, including the sample-then-exclude-parent order. |
| [search_scorer.py](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/evox/utils/search_scorer.py) | Ported `LogWindowScorer`'s reward formula into `agent.window_score`. |
| [controller.py](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/evox/controller.py) | Reference for co-evolution, operator preparation, validation/deployment, population migration, and fallback. |
| [search_strategy_evaluator.py](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/evox/database/search_strategy_evaluator.py) | Adapted validation ideas: singleton/current populations, valid parent/inspiration IDs, bounded context count, and preservation of evaluation data. |
| [search_strategy_db.py](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/evox/database/search_strategy_db.py) | Compared upstream greedy meta-selection with the paper's score-biased, state-conditioned selection. |
| [variation_operator_generator.py](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/search/evox/utils/variation_operator_generator.py) and [search_evolution_user_message.txt](https://github.com/skydiscover-ai/skydiscover/blob/0d932b690670a7e544388ad362e9876c8bd256a0/skydiscover/optimize/context_builder/evox/templates/search_evolution_user_message.txt) | Rewritten as local Slick templates with an explicit generic artifact/strategy interface and JSON contracts. |

Adapted source is covered by SkyDiscover's Apache 2.0 license, reproduced in
[UPSTREAM_LICENSE](UPSTREAM_LICENSE). Upstream copyright: 2025 SkyDiscover Team.
The uniform sampler and scorer were modified for this implementation's smaller
interface, injected RNG, score direction, and actual observed window length.

Deliberate implementation choices:

- Fixed, non-overlapping windows follow the paper's Algorithm 1. The prose calls
  these sliding windows; the inspected controller instead triggers after a
  consecutive run of insignificant per-step improvements. We do not silently
  equate those policies.
- Default strategy reward uses upstream's `1 + log1p(max(0, start))` weight.
  Equation 2 uses `log1p(start)`. A callback exposes the choice; the final partial
  window uses its actual number of steps for normalization.
- Meta-selection uses rank-biased sampling from scored deployments, plus strong
  and similar-state inspirations, following Section 4.3. Upstream selects the
  highest-scoring strategy and random inspirations. The paper does not specify
  a probability distribution or descriptor distance; this implementation uses
  reciprocal-rank weights and normalized absolute descriptor distance.
- Generated programs implement `select(population, state, rng)` rather than
  subclassing SkyDiscover's database. The host owns the append-only solution
  database; evolution retains arbitrary Python selection/operator logic but
  cannot rewrite storage or discard the host population through the interface.
- Operator prose and candidate prompts are intentionally rewritten to support
  arbitrary artifacts. Full artifact generation replaces code-specific patches.
  Refinement, structural variation, free variation, initialization, operator
  preparation, and strategy mutation each own a decorated method and template.
- Execution is injected. The optional worker separates generated strategy code
  from optimizer state; it is not a security sandbox. Task evaluation is entirely
  caller-owned. No benchmark-specific code, model names, or infrastructure from
  SkyDiscover is embedded in the optimizer.

Offline tests exercise behavior and generated-output contracts with the shared
scripted provider plus real subprocess strategy execution. They do not establish
optimization performance or reproduce the paper's reported results.
