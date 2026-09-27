# AlphaEvolve

The default `alphaevolve.paper` implementation uses a persistent MAP-Elites/island
population, multiple maximized metrics, evaluation feedback, and overlapping
generation/evaluation. `Run` executes policies in the existing sandbox. Slick
handles model calls and Pydantic validates generated responses.

## Paper implementation

This independently implements the mechanisms in
[AlphaEvolve, sections 2.1–2.6](https://arxiv.org/html/2506.13131v1#S2).
It is not DeepMind's unpublished implementation, and passing the software tests
does not reproduce the paper's performance results.

| Paper section | Implementation |
| --- | --- |
| 2.1 Task specification | Evaluated initial policies, explicit task/context, protected evolution regions |
| 2.2 Prompt sampling | Population-derived parents/inspirations, probabilistic prompt variants, rendered evaluation output, persistent scored meta-prompt pool |
| 2.3 Generation | Weighted provider ensemble, exact sequential edits, optional full rewrites |
| 2.4 Evaluation | Multiple maximized metrics, threshold cascades, optional LLM feedback, parallel isolated Gym episodes |
| 2.5 Evolution | Persistent evaluated-program database; metric-specific elites within descriptor cells on each island |
| 2.6 Pipeline | Bounded asynchronous generation workers overlapping batched evaluation |

Section 2.5 describes the combination of MAP-Elites and islands without publishing
the exact archive, selection, or migration algorithms. The following are **local,
explicit choices**, not hyperparameters recovered from the paper:

- Each island retains the winner of every metric in every descriptor bin. A
  candidate can remain a parent despite a lower primary score. Ties preserve the
  incumbent. The global primary-score best is retained separately.
- Choose an occupied island uniformly. With probability `exploration` (0.3),
  sample uniformly from its retained members. Otherwise choose a metric uniformly
  and sample with descending rank weights from the top `elite_fraction` (0.2).
  Thus increasing this fraction **broadens the eligible exploitation pool**.
- Sample inspirations across islands, prioritizing different descriptor bins.
- Every `migration_interval` (100) accepted proposals, attempt `migration_count`
  (1) member transfers. A transfer only replaces target cells it improves; it
  never erases an island's whole population.
- AST-equivalent source, including renamed or comment-only copies, shares the
  first canonical evaluation. This is syntactic deduplication, not proof of
  behavioral equivalence. Differently written controllers may behave identically.
- Descriptor bounds/bin counts are supplied by the task. Out-of-range finite
  values clamp to edge bins. Empty descriptor configuration gives one niche per
  metric; configure descriptors for meaningful diversity.

The Gym example uses mean reward, worst-seed reward, and negative reward standard
deviation as maximized metrics. Its descriptors are mean reward and reward
standard deviation. These are **performance niches, not gait descriptors**. The
default ranges are local choices; use multiple seeds to measure variability:

| Environment | Mean reward `(low, high, bins)` | Reward standard deviation |
| --- | --- | --- |
| CartPole | `(0, 500, 20)` | `(0, 250, 10)` |
| LunarLander | `(-500, 350, 34)` | `(0, 500, 10)` |
| BipedalWalker | `(-200, 350, 22)` | `(0, 200, 10)` |

```sh
.venv/bin/python -B -m examples.alphaevolve --env BipedalWalker-v3 --seeds 0 1 2 3 4 5 6 7 8 9
```

`paper` is the default variant. BipedalWalker defaults to complete rewrites;
other environments default to diffs. `--mode diff` or `--mode rewrite` overrides
the choice. Every generated or repaired program must define exactly one top-level
`Solution` class. This prevents a later class from silently replacing the intended
controller; it does not prove that all statements affect the returned action.

`--generations * --batch-size` is the proposal budget for this pipeline, not a
generation barrier. The first batch seeds selection; subsequent generation and
evaluation overlap with bounded pending work. `--generation-concurrency` controls
both model proposals and concurrent runtime repairs, sharing one limit;
`--concurrency` controls sandbox episode workers. Independent failed policies are
repaired concurrently in all variants, with each policy retaining its repair
budget. Duplicate policies share one repair, and cancellation or provider failure
cancels and drains sibling repairs before saving the final search state.
`--batch-size` also limits evaluation batches. Timing can affect search order.

`--search-config config.json` supplies `paper.Config` fields, for example:

```json
{
  "islands": 8,
  "exploration": 0.4,
  "elite_fraction": 0.3,
  "migration_interval": 200,
  "migration_count": 1,
  "meta_interval": 25,
  "features": {
    "mean_reward": [-200, 350, 22],
    "reward_std": [0, 200, 10]
  }
}
```

The Gym adapter supports those two descriptors. Custom evaluators can supply other
numeric descriptors through the Python API. Migration does not inject fresh
founders; no promise is made that these settings escape a particular plateau.

The primary `--model` has weight 1. Add `--ensemble MODEL WEIGHT` repeatedly for
other providers. These model choices are experimental inputs; the default
OpenRouter model does not recreate the paper's Gemini ensemble. Optional
`--initial-policy path.py` evaluates that policy before seeding the archive.
Optional `--screening-seeds 0 1 --screening-min-reward -100` rejects candidates
below that mean reward before evaluating the full seed set. Screening scores are
cached but rejected/partial evaluations never enter the breeding archive.

## Paper Python API

### Resume a paper run

```sh
.venv/bin/python -B -m examples.alphaevolve --resume runs/YOUR_RUN --generations 25
```

This continues in the same directory from `population.sqlite`, loading the resolved
configuration, environment, models, screening settings and evaluation seeds from
`experiment.json`. The original config file and initial-policy source are not
needed or replayed. Existing history and logs are appended; `experiment.json`
remains unchanged.

`--generations` specifies **additional** work, multiplied by the saved batch size.
Without that override, the saved generation count is used. You may also override
`--batch-size`, `--generation-concurrency`, and `--concurrency`. Changing task,
model, evaluation, or search settings is rejected to avoid mixing experiments.
Resume restores the last checkpoint's evaluated population, RNG and search
counters; interrupted model calls and unevaluated proposals are not replayed.
Runs made with `original` or `improved` have no optimizer checkpoint and cannot
use this command. Finishing missing evaluations alone remains `Run.resume()`.

### Use the optimizer directly

Configure `slick.prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent` once.
Supply measured results, keeping every metric's direction maximized:

```python
from research.alphaevolve.paper import AlphaEvolve, Config, EvaluationResult

generator = AlphaEvolve(
    "Walk forward without falling",
    provider,
    context=environment.instructions,
    config=Config(features={"forward_distance": (0, 100, 20)}, mode="rewrite"),
    database_path=run.path / "population.sqlite",
)
try:
    policies = await generator.generate(n=10)
    # The caller's isolated evaluator measures these values.
    results = await evaluate_policies(policies)
    generator.update_results(
        {
            policy.id: EvaluationResult(
                metrics=results[policy.id].metrics,
                features={"forward_distance": results[policy.id].distance},
                feedback=results[policy.id].diagnostic,
            )
            for policy in policies
        }
    )
finally:
    generator.close()
```

`register_initial(policy, result)` seeds all islands with a caller-evaluated policy;
pass `island=` for one island. `search(generator, evaluate_batch, proposals=...)`
runs the overlapping controller, where `evaluate_batch` returns
`{policy.id: EvaluationResult}`. It handles bounded policy repairs and propagates
infrastructure/provider failures. Evaluators own isolation; the optimizer never
executes source itself. The bundled source contract remains `Solution(Policy)`;
other tasks can place their algorithms in that program and supply an evaluator.

For optional staged evaluation and an actual model-based feedback grader:

```python
from research.alphaevolve.paper import EvaluationStage, LLMFeedback, evaluate_cascade

result = await evaluate_cascade(
    policy,
    [EvaluationStage(smoke_test, {"validity": 1}), EvaluationStage(full_evaluation)],
    feedback_evaluator=LLMFeedback(
        feedback_provider,
        {"simplicity": "Grade maintainability from 0 to 1; higher is simpler."},
    ),
)
```

Stage thresholds are minimum values. Later stages may replace earlier estimates;
rejection stops the cascade. The grader may add named rubric metrics or reject a
program, but cannot overwrite measured metric values. Model feedback is optional
and incurs extra calls. Preserve the same complete metric and descriptor schemas
for all accepted programs. Keep held-out evaluations out of training feedback.

`population.sqlite` stores unique evaluated source, metrics, descriptors, outputs,
lineage, cells, and optimizer checkpoints. Reopening with the same archive
configuration and task/context restores selection, RNG, counters, recent failure
context, and scored prompt ideas. The CLI also saves attempt/revision history and
island member IDs to `run.sqlite`. Reopening the population does **not** resume
in-flight model calls or unevaluated proposals, nor reproduce remote model output.
The caller must retain the same evaluator and seed set. `Run.open` alone only
opens evaluation storage; it does not create an optimizer. Full raw model responses
remain in the current process's `attempts`, not a durable transcript; checkpoints
retain only recent rejection excerpts needed for prompts.

## Historical baselines

The optimizer lives in `research/alphaevolve/`, separate from the common `rsikit/` library.
RSIKit owns policies, execution, environments, and run storage. The two historical
baseline variants use that same core:

| Variant | Initial island population | Model feedback |
| --- | --- | --- |
| `alphaevolve.original` | Best initial policy can found every island | Scalar score |
| `alphaevolve.improved` | Separate founders; identical implementations cannot found multiple islands | Scalar score and per-seed rewards |

`original` preserves the local behavior and prompts from commit `5ba5685`.
`improved` preserves the changes introduced in `8a6bc96`. Neither baseline is the CLI default.
Generation, repair budgets, scalar ranking, and evaluation are shared. Each variant
owns its mutation, rewrite, and search-guidance prompts; initialization and repair
prompts are identical and inherited from the original. Configure the common template
root once, as below; selecting the class selects its prompts too.

## Comparing the variants

```bash
.venv/bin/python -B -m examples.alphaevolve --variant original --env LunarLander-v3 --generations 10 --batch-size 25 --seeds 0 1 2 3 4 5 6 7 8 9 --search-seed 0
.venv/bin/python -B -m examples.alphaevolve --variant improved --env LunarLander-v3 --generations 10 --batch-size 25 --seeds 0 1 2 3 4 5 6 7 8 9 --search-seed 0
```

Each command creates a separate Run. Its name includes the variant; `experiment.json`
records CLI arguments, and `run.log` identifies the optimizer class. `--search-seed`
controls optimizer sampling; `--seeds` controls evaluation episodes. Provider output
is still stochastic, so repeat searches instead of treating one pair as conclusive.
Use the same model, environment configuration, search seed sets, proposal budgets,
and repair limits, then evaluate selected policies on held-out episode seeds.

Both variants already use islands. The hypothesis being tested is whether
**independent island founding plus per-seed feedback improves LLM-guided program
search**, not whether introducing islands improves AlphaEvolve. `original` is our
local baseline, not DeepMind's internal implementation. A comparison of these two
variants measures their combined effect; attributing gains to either change needs
islands-only and feedback-only ablations. No performance improvement is established
by this code split. Equal proposal counts also do not guarantee equal model-token
costs or successful evaluation counts; repairs and longer feedback affect those.

## Stored experiment history

All CLI variants save optimizer-owned records in the Run's existing `run.sqlite`:

- `alphaevolve_evaluation`: generation, proposal attempt, policy revision, selected
  island slot, policy ID, parent ID, status, aggregate score, repair count, and error.
- `alphaevolve_generation`: optimizer variant, evaluation seeds, completion flag,
  every island's champion ID and score, and reset events with donor island, target
  island, founder policy, and the evaluated-attempt count at the reset.

These SQLModel classes live in `research/alphaevolve/history.py`. The core only supplies
`run.save(*records)`, which infers each record's table; another optimizer can
supply its own SQLModel records and fields.
Individual episode scores remain in the existing `policy.scores` column. Join by
policy ID instead of relying on generated names, which can repeat.

The CLI commits proposal metadata before sandbox dispatch, failed outcomes before
runtime repairs, replacements before their evaluation, and final results and island
snapshots after each generation. A runtime replacement gets a new revision of the
same attempt, preserving its failed predecessor. Syntax repairs before a valid
policy exists count toward repairs but do not create extra policy revisions.
Repair counts are cumulative within an attempt: take the latest revision per
attempt when totaling repairs, rather than summing every revision's counter.
Discarded proposals may have no policy ID or score; they are included in history.

Provider failures and cancellation save the current generation as incomplete.
A hard process kill can lose updates since the last committed boundary; it cannot
create a completed-generation snapshot. Successfully saved episode scores remain
available even when optimizer selection has not run. For baselines, `complete` means the outer
loop reached `update`, including generations with no surviving policies. Appending
another `run_search` call starts after the last saved generation number. This does
not itself restore optimizer state. The paper pipeline assigns proposals to the
snapshot group when first observed, with island snapshots at evaluated-batch
boundaries. These are asynchronous history groups, not synchronized generations.
Paper snapshots also include `member_count` and member IDs; the `resets` JSON
contains migration events with `source`, `target`, and `policy_id` instead of
baseline reset/donor events.

For island curves, use generation snapshots: the original baseline broadcasts its
initial proposals across all islands even though their proposal slot is 0. A slot
also survives reseeding, so use reset events to distinguish island slots from
uninterrupted lineages. The snapshots are taken after that generation's resets.

For example, this query produces one row per island and completed generation:

```sql
SELECT g.number AS generation,
       json_extract(i.value, '$.island') AS island,
       json_extract(i.value, '$.policy_id') AS policy_id,
       json_extract(i.value, '$.score') AS score
FROM alphaevolve_generation AS g, json_each(g.islands) AS i
WHERE g.complete = 1
ORDER BY generation, island;
```

Use `Evaluation` and `Generation` with `sqlmodel.select`, or query SQLite directly
after the run. Historical runs made before this change have no island history to
backfill reliably. For baselines, calling `generate`/`update` directly writes nothing;
the paper variant persists its archive on `update_results`. The example's outer
loop owns attempt and snapshot history persistence. Token usage and model costs are not
recorded by these tables.

## Baseline Python API

The following uses the improved version. Import `AlphaEvolve` from
`alphaevolve.original` to use the baseline with the same loop. Both accept
`seed_scores`; the original keeps them as data but excludes them from model prompts.
Unless stated otherwise, the founding and feedback behavior below describes the
improved version.

```python
from pathlib import Path

import gymnasium as gym
from slick import prompts
from slick.providers import OpenRouterAPI

from research import alphaevolve
from rsikit import Executor, Run
from research.alphaevolve.improved import AlphaEvolve

# Configure Slick once at application startup.
prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent
generator = AlphaEvolve(
    task="Balance CartPole-v1 for as many steps as possible.",
    context=(
        "Observation: cart position, cart velocity, pole angle, angular velocity. "
        "Action 0 pushes left; 1 pushes right. Each surviving step earns 1 reward."
    ),
    provider=OpenRouterAPI(model="openai/gpt-oss-120b:nitro", max_output_tokens=8192),
)
executor = Executor(concurrency=4)

# Inside an async function:
with gym.make("CartPole-v1", max_episode_steps=500) as environment:
    async with Run.create(name="cartpole", environment=environment, executor=executor) as run:
        for generation in range(25):
            policies = await generator.generate(n=10, concurrency=4)
            scores = await run.evaluate(policies)
            generator.update(scores, seed_scores={p.id: run.scores(p) for p in policies})
        print(generator.best.name)
```

`n=10` means ten proposal attempts in that generation, not a fixed archive size.
Unrepairable candidates are discarded, so the returned list can be shorter or empty.
They are not replaced with extra generation calls.
Founding proposals are generated from the task and distributed across empty islands.
Once all islands have founders, later batches mutate or rewrite evaluated parents.
Names and one-sentence approach descriptions come from the model. Environment instructions are static inputs
supplied by the executor when it creates a policy. The model does not generate or
configure them. Generated policies inherit the constructor and initialize their
own state in `reset()`. Generation runs up to four proposals concurrently and does not execute
policies, create files, or access Run. Every proposal in a batch sees the previous
updates; selection changes only when you call `update`.

`generate(n=10, concurrency=4)` limits concurrent proposal chains, including their
repair and optional guidance calls. Use `concurrency=1` for sequential generation.
Names and descriptions are logged as proposals finish; the returned list preserves
proposal order among survivors. Invalid candidates are discarded without cancelling
siblings. On provider failure or cancellation, unfinished siblings are cancelled and
awaited; that interrupted batch does not enter pending optimizer state. With optional meta
guidance enabled, response timing
can affect which guidance later proposals use. Keep the generate/evaluate/update
loop sequential so every generation uses the previous generation's measured scores.

`evaluate` is the first persistence boundary: it stores and exports the requested
policies before dispatch so interruptions can be resumed, then saves scores and
artifacts as they arrive. It returns `{policy_id: score}`. Identical policy IDs are
evaluated once; their score applies to all matching proposals in `update`.

By default, evaluation uses one episode with seed 0 for every policy. A seed controls
random starting conditions; using the same seed makes comparisons reproducible.
For a broader comparison, `await run.evaluate(policies, seeds=[0, 1, 2])` returns the
mean reward. `run.scores(policy)` retains the individual scores. Use the same seed
set throughout a search and separate held-out seeds when checking generalization.
The optimizer's constructor `seed` controls parent/model selection, separately from
these environment and policy episode seeds.

`update(scores, seed_scores={policy_id: {seed: reward}})` optionally retains per-seed
results alongside each candidate's scalar score. The CLI supplies Run's existing
scores for the current search seeds. Mutation, rewrite, and optional search-guidance
prompts receive this evidence; inspiration policies include their per-seed results too.
Only the scalar score controls selection. Passing `update(scores)` alone remains
supported and supplies no per-seed detail. Keep held-out results out of this feedback.

Pass one configured, serializable environment. The executor loads independent state
for each evaluation process inside a shared Docker container. See [runs](RUNS.md)
for recording, artifacts, lifecycle, and evaluation recovery.

## Baseline search behavior

`Config` controls island count (4), inspirations (3), exploration (0.2), island
reset interval (100 evaluated proposals), optional meta-prompt interval (0 means
disabled), mutation mode (`"diff"` or `"rewrite"`), generation timeout, and `max_repairs` (2 model repair calls per proposal).

The scalar-score archive keeps one champion per island, preserving incumbents on
ties. Founding attempts are distributed round-robin across empty islands; each
candidate competes only on its assigned island. Identical implementations cannot
found multiple islands, even under different generated names. Failed or duplicate
founders leave slots empty for initialization in later batches. This separates
founding lineages but does not guarantee different behavior or algorithm families.
Selection chooses an occupied island's parent;
exploration can use another island's champion. Other distinct champions provide
inspiration. The weaker half of islands is periodically reseeded from survivors,
adapting FunSearch's reset policy (see [NOTICE](../research/alphaevolve/NOTICE) and
[LICENSE.funsearch](../research/alphaevolve/LICENSE.funsearch)). Resets wait
until every island has a founder; subsequent reseeding can share champions.

Exact mutations must match uniquely and stay inside optional EVOLVE-BLOCK regions.
Rewrites preserve the immutable skeleton. Syntax and the top-level `Solution` class
are checked before a policy is returned; execution remains in the sandbox.
Weighted provider ensembles and prompt variants remain available. Optional generated
search guidance is rewarded by positive offspring improvement after `update`.

This API uses one scalar score. The former optimizer-owned evaluation cascade,
multiobjective/diversity result contract, and monolithic `run(initial_source, ...)`
API have been removed. Meaningful behavioral diversity descriptors would need an
explicit result contract when added; a scalar reward does not supply them.
This is a local adaptation of the AlphaEvolve search approach, not a reproduction
of DeepMind's internal implementation or benchmark results.

## Failures and recovery

Generation uses the bounded check → repair → recheck pattern from the `main`
branch's `repair_until_valid` / `RepairingProposer`. Malformed JSON, invalid Python,
missing policy classes, and invalid edits receive diagnostic-driven repair through
a dedicated Slick prompt. Every replacement is checked against the original
parent's protected code before acceptance. Returning an unchanged implementation
is rejected. Successful generation never spends a repair call.

`Config(max_repairs=2)` is the default. This is a **total per-proposal budget** across
syntax/schema repairs and later runtime repairs, not a retry allowance per stage.
`generation_calls` and `repair_calls` count them separately. `max_repairs=0` disables
healing: invalid candidates are discarded immediately. Each model call uses the
configured generation timeout. Exhaustion discards only the affected candidate,
retaining its diagnostic and repair attempts in `generator.attempts`. Generation
returns the survivors; runtime `repair()` returns `None` and removes that policy
from pending optimizer state. Provider errors, model-call timeouts, unexpected failures,
and cancellation propagate without being treated as bad policy output.

The CLI also repairs sandbox `PolicyError` failures, including constructor errors,
invalid actions, and policy execution timeouts. The executor finishes the batch,
preserving successful results, and exposes failed-policy diagnostics through
`error.failures`. Infrastructure errors take priority over policy failures and
stop the run without model repairs. The example composes repair explicitly:

```python
policies = await generator.generate(n=10)
scores = {}
while policies:
    try:
        scores = await run.evaluate(policies)
        break
    except PolicyError as error:
        replacements = {
            policy.id: await generator.repair(policy, error.failures[policy.id])
            for policy in {p.id: p for p in policies}.values()
            if policy.id in error.failures
        }
        if not replacements:
            raise
        policies = [
            replacement for p in policies if (replacement := replacements.get(p.id, p)) is not None
        ]
generator.update(scores, seed_scores={p.id: run.scores(p) for p in policies})
```

`PolicyError` is imported from `rsikit.episode`. Run performs evaluation and storage;
it does not call a model. Repaired policies receive their own IDs and are saved
on the next `evaluate`. Existing successful scores are reused. Failed versions stay
in the run with unfinished scores; no low score is invented. Only the repaired,
evaluated version enters the optimizer's archive. Discarded policies never enter
selection, even if some of their episodes succeeded. An empty generation leaves
the archive unchanged and the CLI continues to the next generation.
Run recovery retries stored
versions; it does not silently rewrite them.

Repair diagnostics and progress appear in the terminal and `run.log`. The full raw
repair responses and checks remain in the optimizer's in-memory `attempts` history.

For the two baselines, optimizer state remains in memory. Reopening Run restores policies and evaluation
work; it does not restore the optimizer's pending proposals, RNG, or islands.

## Example

```sh
uv pip install --python .venv/bin/python -e '.[openrouter]'
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
export OPENROUTER_API_KEY='your-key'
.venv/bin/python -B -m examples.alphaevolve
```

The default attempts 250 proposals (`25 * 10`) using
`openai/gpt-oss-120b:nitro`. For a short run (use `--max-repairs 0` to disable healing):

```sh
.venv/bin/python -B -m examples.alphaevolve --generations 2 --batch-size 3 --max-steps 100
```

`--generation-concurrency` controls concurrent proposals (default 4).
`--concurrency` separately controls sandbox evaluations (default 4).

Select a harder environment with `--env LunarLander-v3` (continuous actions and
wind) or `--env BipedalWalker-v3` (normal terrain). Install `.[openrouter,box2d]`
using uv and rebuild the Docker image first; see [setup commands](../README.md#alphaevolve).
Native episode limits apply unless `--max-steps` is supplied. `--seeds 0 1 2`
averages each policy over three episodes; the default remains seed 0.
These CLI tasks put their description and actual space definitions on a Gymnasium
wrapper's `instructions` attribute. AlphaEvolve receives that text as context, and
the executor supplies the same text to the policy. The generator never invents it.

Run automatically exports every evaluated policy under `exports/` and stores scores
in SQLite. No explicit policy-file writes are needed.

The example shows Rich progress bars for generations, policy generation, and
evaluation. Each completed model response immediately prints the policy name and
description; each saved evaluation immediately prints its score. This is streaming
completion feedback, not token-by-token model output. Each generation ends with a
score table and the best policy so far.

The same log messages and failure tracebacks are saved in `run.log` inside the run
directory. Failed policies retain unfinished scores, and successful evaluations
remain saved. Every model repair is logged with its diagnostic and position in the repair budget.
Library code uses Python's `logging` under `rsikit` and `research.alphaevolve`; the example configures Rich
and file handlers. Policy descriptions are stored in SQLite alongside names.
