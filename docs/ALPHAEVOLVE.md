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
            policies = await generator.generate(n=10)
            scores = await run.evaluate(policies)
            generator.update(scores)
        print(generator.best.name)
```

`n=10` means ten new proposals in that generation, not a fixed archive size.
The first batch is generated from the task. Later batches mutate or rewrite evaluated
parents. Names come from the model. Environment instructions are static inputs
supplied by the executor when it creates a policy. The model does not generate or
configure them. Generated policies inherit the constructor and initialize their
own state in `reset()`. Generation is sequential and does not execute
policies, create files, or access Run. Every proposal in a batch sees the previous
updates; selection changes only when you call `update`.

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
disabled), mutation mode (`"diff"` or `"rewrite"`), and generation timeout.

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

Generation has no hidden retry loop. Malformed output, invalid edits, provider
failures, and generation timeouts raise and are recorded in `attempts`, including
raw responses. A failed batch returns no policies and is not registered for update.
Unexpected errors and cancellation also propagate. Optional meta-prompt failure
falls back to existing guidance without retrying that meta call.

Evaluation errors preserve successful scores and leave unfinished evaluations
available to `Run.resume()`. After recovery, evaluating the same batch reads its
stored scores, which can then be passed to `update`. `update` accepts only pending
policy IDs and finite scores, and consumes each pending proposal once.

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
`openai/gpt-oss-120b:nitro`. For a short run:

```sh
.venv/bin/python -B -m examples.alphaevolve --generations 2 --batch-size 3 --max-steps 100
```

Run automatically exports every evaluated policy under `exports/` and stores scores
in SQLite. No explicit policy-file writes are needed.
