# Introduction and concepts

rsikit is a Python toolkit for generating policies, evaluating them in Gymnasium
environments, and composing optimization loops with saved evidence. It is for
researchers and developers who want to compare executable policies or build their
own search strategy.

The shared `rsikit` package supplies policy, evaluation, execution, optimization,
and storage contracts. The packaged `research` modules implement particular search
algorithms. Those implementations are experimental adaptations, not guarantees of
reproducing a paper's results. In this project, “self-improvement” means using
measured outcomes to revise policy source, prompts, or search strategies; it does
not train or modify the underlying language model's weights.

## Vocabulary

| Term | Meaning |
| --- | --- |
| Environment | A Gymnasium object defining observations, valid actions, transitions, rewards, and episode endings. |
| Task | Instructions describing the environment and objective to the policy generator or optimizer. |
| `Policy` | A live Python object with asynchronous `reset`, `act`, and `close` methods. |
| `PolicyDefinition` | An immutable source document, with a name, description, and content identity. It is not a live policy. |
| Episode | The trajectory from one reset: observations, actions, rewards, termination/truncation flags, infos, and possibly an error and artifacts. |
| Evaluation | Executing a policy on one or more seeds to produce episodes. |
| Optimizer | An object proposing definitions and accepting episode feedback until its budget or stopping rule is met. |
| Proposal round | One `propose()` call followed by one complete `update(results)` call. |
| Run | A directory and database holding policies, episodes, scores, and application-specific records. |

## The loop

```text
Optimizer.propose() → PolicyDefinitions
                            ↓
                Evaluate each on a seed panel
                            ↓
                  {policy_id: {seed: Episode}}
                            ↓
                   Optimizer.update(results)
                            ↓
                 repeat until optimizer.done
                            ↓
                      optimizer.best
```

`search` drives this loop. Your evaluation callback chooses the environment,
seeds, workers, and persistence; the optimizer chooses candidates and selection.
A round may contain one candidate or many. Algorithms use different budgets and
may make extra model calls for repair or reflection, so their “generations” are
not equal units of work.

Each transition has an environment reward. `Episode.total_reward` sums them.
The shared scoring helpers use the last info's `fitness` value when present,
otherwise total reward. Fitness must be a finite number. Optimizers usually
aggregate per-seed fitness into a mean; some use additional objectives. A failed
episode is diagnostic evidence, never a successful score.

Start with [installation](installation.md), then follow either
[an existing system](../examples/existing-system.md) or
[a custom optimization system](../examples/custom-system.md). The
[API reference](../api/index.md) defines the exact contracts.
