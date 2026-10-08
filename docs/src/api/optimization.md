# Optimization contract

`Optimizer` describes a complete-round interface: `propose()` returns definitions,
then `update(results)` consumes episodes keyed by exactly those policy IDs. The
optimizer owns selection; the evaluator owns execution. `search` connects the two
without creating environments, clients, workers, or a run directory.

See the [optimization guide](../guide/optimization.md) for a search loop and
[concrete optimizers](optimizers.md) for the implementations in `research`.
The [custom system example](../examples/custom-system.md) shows how to implement
this contract for another system.

<!-- api: rsikit.optimization.Optimizer
{members: [done, best, propose, update], inherited_members: false}
-->

<!-- api: rsikit.optimization.search
{}
-->

## Validating feedback

Custom optimizers can use `validate_results` before modifying their state. Keep
the returned seed panel and pass it to later calls when successful candidates
must use the same seeds across rounds. This validates episode structure and seed
alignment; score extraction is handled by
[`episode_scores` and `episode_error`](evaluation.md#optimizer-helpers).

<!-- api: rsikit.optimization.validate_results
{}
-->

