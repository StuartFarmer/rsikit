# AlphaEvolve

AlphaEvolve generates named `Policy` classes in memory. `Run` evaluates and stores
them. Scores go back to AlphaEvolve to guide the next batch. Slick handles model
calls and Pydantic validates generated responses; there is no slick-bits dependency.

```python
from pathlib import Path

import gymnasium as gym
from slick import prompts
from slick.providers import OpenRouterAPI

import rsikit.alphaevolve as alphaevolve
from rsikit import Executor, Run
from rsikit.alphaevolve import AlphaEvolve

# Configure Slick once at application startup.
prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent / "prompts"
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
            generator.update(scores)
        print(generator.best.name)
```

`n=10` means ten new proposals in that generation, not a fixed archive size.
The first batch is generated from the task. Later batches mutate or rewrite evaluated
parents. Names and one-sentence approach descriptions come from the model. Environment instructions are static inputs
supplied by the executor when it creates a policy. The model does not generate or
configure them. Generated policies inherit the constructor and initialize their
own state in `reset()`. Generation runs up to four proposals concurrently and does not execute
policies, create files, or access Run. Every proposal in a batch sees the previous
updates; selection changes only when you call `update`.

`generate(n=10, concurrency=4)` limits concurrent proposal chains, including their
repair and optional guidance calls. Use `concurrency=1` for sequential generation.
Names and descriptions are logged as proposals finish; the returned list preserves
proposal order. On failure or cancellation, unfinished siblings are cancelled and
awaited; a partial batch does not enter pending optimizer state. With optional meta
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

Pass one configured, serializable environment. The executor loads independent state
for each evaluation process inside a shared Docker container. See [runs](RUNS.md)
for recording, artifacts, lifecycle, and evaluation recovery.

## Search behavior

`Config` controls island count (4), inspirations (3), exploration (0.2), island
reset interval (100 evaluated proposals), optional meta-prompt interval (0 means
disabled), mutation mode (`"diff"` or `"rewrite"`), generation timeout, and `max_repairs` (2 model repair calls per proposal).

The scalar-score archive keeps one champion per island, preserving incumbents on
ties. Initial results can found every island. Selection chooses an island's parent;
exploration can use another island's champion. Other distinct champions provide
inspiration. The weaker half of islands is periodically reseeded from survivors,
adapting FunSearch's reset policy (see NOTICE and LICENSE.funsearch).

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
healing. Each model call uses the configured generation timeout. Exhaustion raises
`InvalidCandidate` with the final diagnostic, retaining the repair attempts in
`generator.attempts`. Provider errors, model-call timeouts, unexpected failures,
and cancellation propagate without being treated as bad policy output.

The CLI also repairs sandbox `PolicyError` failures, including constructor errors,
invalid actions, and policy execution timeouts. The executor finishes the batch,
preserving successful results, and exposes failed-policy diagnostics through
`error.failures`. Infrastructure errors take priority over policy failures and
stop the run without model repairs. The example composes repair explicitly:

```python
policies = await generator.generate(n=10)
while True:
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
        policies = [replacements.get(p.id, p) for p in policies]
generator.update(scores)
```

`PolicyError` is imported from `rsikit.episode`. Run performs evaluation and storage;
it does not call a model. Repaired policies receive their own IDs and are saved
on the next `evaluate`. Existing successful scores are reused. Failed versions stay
in the run with unfinished scores; no low score is invented. Only the repaired,
evaluated version enters the optimizer's archive. Run recovery retries stored
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
Library code uses Python's `logging` under `rsikit`; the example configures Rich
and file handlers. Policy descriptions are stored in SQLite alongside names.
