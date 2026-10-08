# Evaluation and episodes

Use `evaluate` for caller-owned objects in the current process. It imposes no
wall-clock timeout and does not close either input. Use [execution](execution.md)
for worker processes and a deadline.

<!-- api: rsikit.evaluation.evaluate
{}
-->

<!-- api: rsikit.evaluation.Evaluator
{members: [evaluate], inherited_members: false}
-->

## Episode data

`Episode` stores a trajectory and optional failure diagnostic. A failed episode's
rewards describe partial execution and should not enter successful selection.
The final `info["fitness"]` can supply a score separate from the reward sum.

<!-- api: rsikit.episode.Episode
{members: [observations, actions, rewards, terminations, truncations, infos, artifacts,
    error, total_reward, final_step, validate_complete, encode, from_data, decode],
  inherited_members: false}
-->

## Optimizer helpers

These helpers support [custom optimizers](optimization.md). `episode_scores`
omits failed seeds; use `episode_error` to reject a partly failed panel before
comparing it with successful candidates.

<!-- api: rsikit.evaluation.episode_scores
{}
-->

<!-- api: rsikit.evaluation.episode_error
{}
-->

## Exceptions

Import these from `rsikit.evaluation`. Direct evaluation records policy failures
in `Episode.error`; environment and infrastructure errors can propagate.
`Run.mean_scores` raises `PolicyError` for failed panels and supplies diagnostics
through its `failures` mapping. Worker timeouts are returned as episode errors;
callers should not rely on receiving a `PolicyTimeout` exception.

<!-- api: rsikit.evaluation.PolicyError
{members: [], inherited_members: false}
-->

<!-- api: rsikit.evaluation.PolicyTimeout
{members: [], inherited_members: false}
-->

<!-- api: rsikit.evaluation.InfrastructureError
{members: [], inherited_members: false}
-->

