# AlphaEvolve

`rsikit.alphaevolve` evolves Python `Solution(Controller)` classes with direct Slick
provider calls and Pydantic response models. It adapts the local `slick-bits`
reference implementation; that repository is neither imported nor installed.
This is an implementation of the search approach, not a reproduction of
DeepMind's internal system or its reported benchmark results.

## Use

```python
from functools import partial
from pathlib import Path

from slick import prompts
from slick.providers import OpenAIAPI

import rsikit.alphaevolve as alphaevolve
from rsikit.alphaevolve import AlphaEvolve, Config, evaluate_program

# Configure Slick once at application startup, before concurrent calls.
prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent / "prompts"

agent = AlphaEvolve(
    task="Balance the CartPole pole for as many steps as possible.",
    context=(
        "Observation: cart position, cart velocity, pole angle, angular velocity. "
        "Action 0 pushes left; 1 pushes right. Each surviving step earns 1. "
        "Evaluation averages seeds 0, 1, 2 with a 200-step cap."
    ),
    provider=OpenAIAPI(model="YOUR_MODEL", max_output_tokens=4096),
    evaluate=partial(evaluate_program, make_env="CartPole-v1", seeds=(0, 1, 2), max_steps=200),
    config=Config(islands=4),
)

# Inside an async function; read source as data, never import generated programs.
best = await agent.run(Path("solution.py").read_text(), attempts=100, concurrency=4, seed=7)
Path("best.py").write_text(best.content)
print(best.metrics)
```

The seed must export a top-level `Solution` class. The same policy contract and
spaces apply as in the [inner loop](INNER_LOOP.md). The default evaluator accepts
an existing Gymnasium ID or an environment factory, runs a fresh Docker episode
for every fixed seed, and maximizes mean return under the `reward` metric.
Its feedback includes episode lengths, termination flags, and environment info.
An empty seed list is not a valid evaluation. Use separate held-out seeds to
assess generalization after search.

Install RSIKit with `.[openai]` for the example provider, or pass another Slick
provider supported by your application. The bundled CLI example is
`python -m examples.alphaevolve --model YOUR_MODEL`. It runs sequentially
and saves the best source. Docker is required for `evaluate_program`; no model
credentials or network access are given to generated policies.

## Search mechanisms

- **Islands and selection:** all islands start from the evaluated seed. Sample
  an island and objective; select uniformly from its archive with probability
  `exploration` (default 0.2), otherwise select that objective's champion.
- **Inspirations:** include up to `inspirations` (default 3) distinct other
  programs from the island and global champions. They provide material for
  recombination in the model's proposal, without a separate crossover scheduler.
- **Diversity:** each island keeps a champion per `(cell, objective)`. An injected
  evaluator supplies discrete `cell` descriptors. The default `cell=()` keeps
  one cell; it does not invent meaningful behavior descriptors automatically.
- **Island reset:** every `reset_interval` completed attempts (default 100),
  reseed the weaker half from surviving islands. This policy adapts FunSearch;
  see the bundled NOTICE and Apache license. In-flight offspring join the current
  island when they finish. Global champions survive resets.
- **Model selection:** optional `ensemble=((provider_a, 0.8), (provider_b, 0.2))`
  replaces the single provider. Weights are sampling probabilities, never model
  parameter updates. Optional weighted `prompt_variants` add search guidance.
- **Evaluation cascade:** `stages=(EvaluationStage(cheap_check, {"valid": 1}),)`
  gates expensive evaluation. Only the final evaluator's metrics are objectives.
- **Prompt evolution:** `Config(meta_interval=10)` generates guidance every tenth
  attempt and scores it by subsequent positive offspring improvement. Disabled
  by default; these model calls are additional to the candidate budget.

Each evaluator is `async evaluate(source: str) -> Evaluation`. For custom metrics
or diversity, return `Evaluation(metrics={"quality": 0.9, "negative_cost": -2.0},
feedback="Measured diagnostics", cell=(2, 5))`. All objectives are maximized and
must be finite, nonempty, and have consistent names. `target_metric` on `run`
selects the returned winner and ranks islands for reset. Ties retain incumbents.
This archive stores per-objective champions, not a full Pareto frontier.

## Generated output and editing

Default `Config(mode="diff")` requests JSON with an `edits` list, each containing
`search` and `replacement`. Searches must match exactly once and edits apply
sequentially. `mode="rewrite"` requests JSON with the complete `source` field.
Prompt evolution uses JSON with an `instruction` field. Pydantic validates these
closed response schemas; external Jinja templates explicitly include them.
This JSON contract deliberately replaces the reference's free-text diff format.

Optional `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` comments restrict edits.
Marker lines and text outside the blocks remain unchanged; nested and unbalanced
markers are rejected. Without markers the whole file is editable. Syntax and the
presence of `Solution` are checked without executing source; actual class behavior
is checked by the evaluator. Markers are edit boundaries, not a security sandbox.

## Budgets and failures

`attempts` counts candidate attempts, including rejections; seed evaluation is
additional. `concurrency` bounds in-flight attempts and completed children enter
the archive immediately. A fresh agent owns each run. Use `concurrency=1` for
reproducible selection with scripted responses; parallel completion order affects
sampling. Providers must support concurrent calls when concurrency is enabled.

`generation_calls`, `meta_calls`, and `evaluations` count calls separately.
`evaluations` counts evaluator stages, not the episodes inside an evaluator.
`programs`, `islands`, `best_by_metric`, `attempts`, and reset `events` expose state.
Attempts retain exact raw responses before parsing, including malformed JSON.
History is in memory; callers own durable storage. Slick's template root is
process-global; do not switch it during concurrent operations.

Malformed proposals, invalid edits/programs, provider failures, configured
`generation_timeout` / `evaluation_timeout`, and explicit evaluation rejection
consume an attempt. A failed optional meta call falls back to existing guidance.
There are no automatic agent retries. The default evaluator converts policy
failures into rejection, but propagates Docker/infrastructure and environment
errors. Unexpected evaluator errors abort and cancel sibling workers; a failed
seed aborts before generation. Custom evaluators own execution isolation and can
reject with `Evaluation(error="reason")` or `InvalidCandidate`.

The checks exercise search mechanics with scripted model responses and real
Docker/Gymnasium evaluation. They establish working integration, not model-driven
optimization quality or reproduction of AlphaEvolve's scientific results.
