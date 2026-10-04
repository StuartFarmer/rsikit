# AlphaEvolve

The default `alphaevolve.paper` implementation uses a persistent MAP-Elites/island
population, multiple maximized metrics, evaluation feedback, and concurrent
generation/evaluation. `Executor` evaluates policies; `Run` stores their results. Slick
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
| 2.6 Pipeline | Complete proposal/evaluation rounds with bounded concurrency within each stage; cross-stage overlap is deferred |

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
./scripts/run examples.alphaevolve --env BipedalWalker-v3 --seeds 0 1 2 3 4 5 6 7 8 9
```

`paper` is the default variant. BipedalWalker defaults to complete rewrites;
other environments default to diffs. `--mode diff` or `--mode rewrite` overrides
the choice. Every generated or repaired program must define exactly one top-level
`Solution` class. This prevents a later class from silently replacing the intended
controller; it does not prove that all statements affect the returned action.

`--generations * --batch-size` is the original-attempt budget. Each complete
proposal round is measured before the next round is generated; the first round
seeds selection. `--generation-concurrency` controls
both model proposals and concurrent runtime repairs, sharing one limit;
`--concurrency` controls episode processes. Independent failed policies are
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
./scripts/run examples.alphaevolve --resume runs/YOUR_RUN --generations 25
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
counters, pending evaluation rounds, and queued repairs. Interrupted remote model
calls cannot be replayed.
Runs made with `original` or `improved` have no optimizer checkpoint and cannot
use this command. Controllers request missing episodes explicitly; Run owns storage.

### Use the optimizer directly

Configure `slick.prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent` once.
All three variants implement [the common optimizer contract](INNER_LOOP.md#one-optimization-loop).
Batch size and original-attempt limits belong to `Config`:

```python
from rsikit import Measurement, search
from research.alphaevolve.paper import AlphaEvolve, Config

optimizer = AlphaEvolve(
    "Walk forward without falling",
    provider,
    context=environment.instructions,
    config=Config(
        proposals=250, batch_size=10, mode="rewrite", features={"forward_distance": (0, 100, 20)}
    ),
    database_path=run.path / "population.sqlite",
)


async def evaluate(policies):
    measured = await evaluate_policies(policies)  # Caller-owned isolated evaluation.
    return {
        p.id: Measurement(
            scores=measured[p.id].seed_scores,
            features={"forward_distance": measured[p.id].distance},
            feedback=measured[p.id].diagnostic,
        )
        for p in policies
    }


try:
    best = await search(optimizer, evaluate)
finally:
    optimizer.close()
```

`propose()` returns unique policy definitions. `update()` consumes exactly those
IDs mapped to neutral measurements; raw Episode pairs and scalar mappings are no
longer its public feedback format. Seed identity survives evaluation. Original
and Improved maximize mean seed scores. Paper builds its private `EvaluationResult`
in update, deriving `reward` (mean), `worst_reward` (minimum), `stability` (negative
population standard deviation), and configured `mean_reward`/`reward_std`
descriptors. Explicit measured metrics and features override derived values.
Required objective/descriptor evidence is checked before any archive mutation.

Screening and grading stay in the evaluator. A rejected measurement does not
trigger repair; `Measurement(failure="diagnostic")` queues repair for the next
proposal round. Repairs preserve original attempt counts and successful siblings.
Per-seed variation is retained even when means are equal. With one seed, observed
standard deviation is zero; it does not estimate unseen-seed variability.

`register_initial(policy, result)` remains an AlphaEvolve-specific archive helper
using its own result type, with `island=` selecting one island. The historical
`research.alphaevolve.paper.search(..., proposals=...)` entry point is a thin
wrapper around `rsikit.search`; its evaluator now returns `Measurement`, and its
legacy return remains `None`. New integrations should use the core runner, which
returns the best policy or `None`.

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
island member IDs to `run.sqlite`. Round checkpoints preserve original attempts,
pending policy definitions, seed panels, and queued repairs. Reopening reissues
pending evaluation and continues repairs without allocating another original
attempt. It cannot replay an in-flight remote model call or guarantee identical
remote output. Keep the same evaluator and seeds. Legacy completed archives are
accepted; incomplete legacy streaming checkpoints are rejected before generation.
`Run.open` alone opens evaluation storage and does not restore an optimizer.

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
Generation, repair budgets, and scalar ranking are shared. Each variant
owns its mutation, rewrite, and search-guidance prompts; initialization and repair
prompts are identical; variants share the original `SelfHealer`. Configure the common template
root once, as below; selecting the class selects its prompts too.

## Comparing the variants

```bash
./scripts/run examples.alphaevolve --variant original --env LunarLander-v3 --generations 10 --batch-size 25 --seeds 0 1 2 3 4 5 6 7 8 9 --search-seed 0
./scripts/run examples.alphaevolve --variant improved --env LunarLander-v3 --generations 10 --batch-size 25 --seeds 0 1 2 3 4 5 6 7 8 9 --search-seed 0
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

The CLI commits proposal metadata before episode dispatch, failed outcomes before
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
available even when optimizer selection has not run. For baselines, `complete` means all attempts and repairs settled, including
generations with no surviving policies. Appending
another `run_search` call starts after the last saved generation number. This does
not itself restore optimizer state. Paper now uses synchronized proposal rounds
with island snapshots after feedback and any bounded repairs have settled.
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
from rsikit import Executor, Run, search
from research.rollouts import Rollouts
from research.rewards import measure_rewards
from research.alphaevolve.improved import AlphaEvolve, Config

# Configure Slick once at application startup.
prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent
generator = AlphaEvolve(
    task="Balance CartPole-v1 for as many steps as possible.",
    context=(
        "Observation: cart position, cart velocity, pole angle, angular velocity. "
        "Action 0 pushes left; 1 pushes right. Each surviving step earns 1 reward."
    ),
    provider=OpenRouterAPI(model="openai/gpt-oss-120b:nitro", max_output_tokens=8192),
    config=Config(proposals=250, batch_size=10, generation_concurrency=4),
)
executor = Executor(concurrency=4)

# Inside an async function:
with gym.make("CartPole-v1", max_episode_steps=500) as environment:
    async with executor, Run.create(name="cartpole") as run:
        rollouts = Rollouts(environment, executor, run)
        best = await search(
            generator,
            lambda policies: measure_rewards(rollouts, policies, seeds=(0, 1, 2)),
        )
        if best is not None:
            print(best.name)
```

`batch_size=10` means ten original proposal attempts per round, not an archive
size. Invalid candidates consume their existing repair allowance and can leave
fewer survivors. All-invalid batches advance internally until candidates are
available or the original-attempt budget is exhausted.

Founders are measured before descendant generation. Every batch sees prior
feedback; selection changes only in update. `generation_concurrency` bounds
proposal chains including repairs and guidance calls. Episode concurrency belongs
to the evaluator. Names/descriptions are generated, while environment instructions
remain caller-supplied. Generated policies inherit the constructor and initialize
state in `reset()`.

`measure_rewards` returns per-seed `Measurement` values and persists individual
scores and episodes through Run. Identical returned policy IDs are evaluated once.
Use the same seed set throughout a search and separate held-out seeds when
checking generalization. The optimizer constructor's `seed` controls parent/model
selection, separately from environment and policy episode seeds.

The old `generate(n, concurrency=...)` and `update_scores(scores, seed_scores=...)`
helpers remain for low-level callers. They are outside the common complete-round
contract; new orchestration should use `propose()` and `update()` or `search()`.
No loop should mix these helper conventions with an outstanding public round.

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
These rules and the mutation response schema belong to AlphaEvolve's own
`generation.py`. Generation and repair operations return policy definitions;
the optimizer explicitly calls `validate_policy` before accepting them.
Rewrites preserve the immutable skeleton. Syntax and the top-level `Solution` class
are checked before a policy is returned; execution remains in an episode process.
Weighted provider ensembles and prompt variants remain available. Optional generated
search guidance is rewarded by positive offspring improvement after `update`.

This API uses one scalar score. The former optimizer-owned evaluation cascade,
multiobjective/diversity result contract, and monolithic `run(initial_source, ...)`
API have been removed. Meaningful behavioral diversity descriptors would need an
explicit result contract when added; a scalar reward does not supply them.
This is a local adaptation of the AlphaEvolve search approach, not a reproduction
of DeepMind's internal implementation or benchmark results.

## Failures and recovery

Generation uses a bounded check → repair → recheck loop. The optimizer delegates
model repair to `original.healing.SelfHealer`, a standalone class holding its task,
context, provider, and repair prompt. It uses Slick's functional render/parse API
and records raw responses before parsing. The optimizer owns retries and acceptance.
Malformed JSON, invalid Python, missing policy classes, and invalid edits receive
diagnostic-driven repair. Every replacement is checked against the original
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

The CLI also repairs `PolicyError` failures, including constructor errors,
invalid actions, and policy execution timeouts. The executor finishes the batch,
preserving successful results, and exposes failed-policy diagnostics through
`error.failures`. Infrastructure errors take priority over policy failures and
stop the run without model repairs. The common loop handles repair decisions:

```python
from rsikit import search
from research.rewards import measure_rewards

best = await search(generator, lambda ps: measure_rewards(rollouts, ps, seeds=(0, 1)))
```

The evaluator supplies failed measurements; update queues repairs; the next
proposal round generates replacements. Successful siblings are retained and are
not resubmitted. Failed versions keep their evidence and no low score is invented.
Only accepted, measured versions enter the archive. Discarded attempts consume
the original attempt budget; the optimizer terminates even if no program survives.

Run recovery retries stored
versions; it does not silently rewrite them.

Repair diagnostics and progress appear in the terminal and `run.log`. The full raw
repair responses and checks remain in the optimizer's in-memory `attempts` history.

For the two baselines, optimizer state remains in memory. Reopening Run restores policies and evaluation
work; it does not restore the optimizer's pending proposals, RNG, or islands.

## Example

```sh
export OPENROUTER_API_KEY='your-key'
./scripts/run examples.alphaevolve
```

The default attempts 250 proposals (`25 * 10`) using
`openai/gpt-oss-120b:nitro`. For a short run (use `--max-repairs 0` to disable healing):

```sh
./scripts/run examples.alphaevolve --generations 2 --batch-size 3 --max-steps 100
```

`--generation-concurrency` controls concurrent proposals (default 4).
`--concurrency` separately controls episode evaluations (default 4).

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
