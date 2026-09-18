# AlphaEvolve

AlphaEvolve generates named `Policy` classes in memory. `Run` evaluates and stores
them. Scores go back to AlphaEvolve to guide the next batch. Slick handles model
calls and Pydantic validates generated responses; there is no slick-bits dependency.

The optimizer lives in the top-level `alphaevolve/` package, alongside `rsikit/`.
RSIKit owns policies, execution, environments, and run storage. The two optimizer
variants use that same core:

| Variant | Initial island population | Model feedback |
| --- | --- | --- |
| `alphaevolve.original` | Best initial policy can found every island | Scalar score |
| `alphaevolve.improved` | Separate founders; identical implementations cannot found multiple islands | Scalar score and per-seed rewards |

`original` preserves the local behavior and prompts from commit `5ba5685`.
`improved` preserves the changes introduced in `8a6bc96` and remains the CLI default.
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

Both CLI variants save optimizer-owned records in the Run's existing `run.sqlite`:

- `alphaevolve_evaluation`: generation, proposal attempt, policy revision, selected
  island slot, policy ID, parent ID, status, aggregate score, repair count, and error.
- `alphaevolve_generation`: optimizer variant, evaluation seeds, completion flag,
  every island's champion ID and score, and reset events with donor island, target
  island, founder policy, and the evaluated-attempt count at the reset.

These SQLModel classes live in `alphaevolve/history.py`. The core only supplies
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
available even when optimizer selection has not run. `complete` means the outer
loop reached `update`, including generations with no surviving policies. Appending
another `run_search` call starts after the last saved generation number. This does
not restore optimizer state.

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
backfill reliably. Calling `generate`/`update` directly still writes nothing; the
example's outer loop owns history persistence. Token usage and model costs are not
recorded by these tables.

## Python API

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

import alphaevolve
from rsikit import Executor, Run
from alphaevolve.improved import AlphaEvolve

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
    with Run.create(name="cartpole", environment=environment, executor=executor) as run:
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

## Search behavior

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
adapting FunSearch's reset policy (see [NOTICE](../alphaevolve/NOTICE) and
[LICENSE.funsearch](../alphaevolve/LICENSE.funsearch)). Resets wait
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

Optimizer state remains in memory. Reopening Run restores policies and evaluation
work; it does not restore the optimizer's pending proposals, RNG, or islands.

## Example

```sh
uv pip install --python .venv/bin/python -e '.[openrouter]'
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
export OPENROUTER_API_KEY='your-key'
.venv/bin/python -B -m examples.alphaevolve
```

The default makes 25 generations of 10 proposals using
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
Library code uses Python's `logging` under `rsikit` and `alphaevolve`; the example configures Rich
and file handlers. Policy descriptions are stored in SQLite alongside names.
