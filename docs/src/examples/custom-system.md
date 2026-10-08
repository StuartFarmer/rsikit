# Implement a custom system

Write an optimizer that compares three handwritten policies, evaluates them with
the shared execution pipeline, and exports its best result. This example requires
no provider or API key and makes no model calls.

## Prerequisites and command

Follow the [base clone installation](../guide/installation.md) on macOS or Linux,
then run from the repository root:

```bash
.venv/bin/python -m examples.custom_system --output runs/custom-demo
```

The output directory must be new. Omit `--output` for an automatically named run.
The script usually takes a few seconds and costs nothing. A verified Python
3.14/macOS run finished in under a second and printed:

```text
training_mean=100.0
heldout_mean=100.0
winner=.../runs/custom-demo/best.py
```

The budget is exactly three candidates on two training seeds and a winner on two
held-out seeds: eight episodes of at most 100 steps, using one worker.

## Optimizer and policy contracts

`GridSearch` implements the [`Optimizer` protocol](../api/optimization.md) through
`done`, `best`, `propose()`, and `update(results)`. It proposes one round, then
accepts a complete mapping of policy IDs to seed-to-`Episode` mappings. It checks
IDs, episode validity, and the training seed panel before changing its state.
Repeating a proposal before feedback or submitting feedback twice raises an error.

Selection uses mean fitness, including a final-info `fitness` override when an
environment supplies one. Any candidate with a failed episode or empty feedback
is excluded. A partial trajectory with a high reward cannot win. If all candidates
fail, `best` remains `None` and the script fails visibly.

Each candidate is a `PolicyDefinition` containing a small `Solution(Policy)`.
The policy remembers the previous angle and resets that memory every episode,
after calling `super().reset(seed=seed)`. Its action is always an integer `0` or
`1`, valid for CartPole's discrete action space. The three gains change the
response to recent pole motion. See [policies](../guide/policies.md) for lifecycle
details and [customization](../guide/customization.md) for a custom-environment
recipe.

## Script

Edit `examples/custom_system.py`; this page embeds that same file.

```python
{{#include ../../../examples/custom_system.py}}
```

## Extend and inspect it

`search` drives the proposal/feedback loop. Its evaluation callback delegates to
`Run.evaluate`, which persists episodes and scores through a context-managed
`Executor`. `Run` and the original environment also have explicit context owners.
The optimizer receives raw episodes, not precomputed means.

Replace the candidate list with an LLM proposal strategy when needed, keeping
the same feedback boundary. To add rounds, retain the current best and set `done`
only after the intended budget. This example deliberately has no optimizer
checkpoint or resume implementation; reopening its Run restores saved evidence,
not optimizer state.

The winner is written to `best.py`, reloaded with `PolicyDefinition.from_file`,
and evaluated on seeds `100` and `101`, which never enter training selection.
`exports/` holds all three candidate definitions; `run.sqlite` and `episodes/`
hold measured evidence. See [runs and recovery](../guide/runs.md) for querying
these files. Saved episode data uses pickle and must be trusted.

The offline [smoke checks](https://github.com/StuartFarmer/rsikit/blob/main/tests/test_release_examples.py) run both featured
scripts with real workers, reopen the results, and verify selection, exported
identities, held-out measurements, and feedback failure handling:

```bash
.venv/bin/python -m unittest tests.test_release_examples -v
```
