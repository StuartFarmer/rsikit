# Run storage

Create runs through `Run.create`, or reopen a trusted directory with `Run.open`.
Both lock the directory immediately; use a synchronous or asynchronous context
manager to release resources and capture run progress. See the [run guide](../guide/runs.md)
for the on-disk layout and common workflows.

A run's cache key is **policy ID plus seed**. Use a new run if the environment or
evaluation configuration changes. Successful cached episodes are reused; failed
episodes are retried. Run directories contain pickle data, which can execute
Python when loaded, so only open directories you trust.

`collect` saves raw evidence, `evaluate` also records successful scores, and
`mean_scores` requires every requested episode to succeed. `save_policy` and
`save_episode` support external evaluation without running any jobs themselves.

<!-- api: rsikit.run.Run
{members: [create, open, path, name, close, aclose, save, database, policies, scores,
    save_policy, save_episode, load_episode, collect, evaluate, mean_scores], inherited_members: false}
-->

