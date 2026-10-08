# Optimization

`search(optimizer, evaluate)` drives proposal rounds and returns the optimizer's
best `PolicyDefinition`, or `None` if no candidate succeeded. It creates no
workers and chooses no environment: pass an async evaluation callback that owns
those decisions.

## Complete-round feedback

The callback receives a sequence of definitions and returns episodes:

```python
{
    policy.id: {
        0: episode_for_seed_0,
        1: episode_for_seed_1,
    },
}
```

Return exactly the IDs from the outstanding proposal round, including failed or
rejected candidates. A failed episode carries `error`; an empty seed mapping can
represent a rejected candidate with no accepted episodes. Successful candidates
must have a complete, common seed panel. `search` validates round IDs and episode
structure before calling `optimizer.update(results)`. Built-in optimizers also
check their expected panel across rounds.

Do not pass `Run.mean_scores` as the callback: aggregate scores discard the
evidence required by the optimizer protocol. `Run.evaluate` supplies the correct
shape. The [custom-system example](../examples/custom-system.md) provides a small,
offline implementation of `done`, `best`, `propose`, and `update`.

## Generate one definition

`generate(task, provider=...)` asks a Slick provider for source and metadata. It
does not evaluate the result. Configure the packaged generation templates before
calling it:

```python
from pathlib import Path
from slick import prompts
from slick.providers import OpenRouterAPI
import rsikit.generation as generation
from rsikit import generate

prompts.TEMPLATE_ROOT = Path(generation.__file__).parent / "prompts"
provider = OpenRouterAPI(model="YOUR_OPENROUTER_MODEL_ID")
# Inside an async function, with OPENROUTER_API_KEY configured:
policy = await generate("Write a CartPole-v1 controller maximizing reward.", provider=provider)
policy.validate()
```

Template roots are shared Slick configuration; the unified CLI adapters set and
restore their own roots. For a complete paid workflow, use
[the existing-system walkthrough](../examples/existing-system.md).

## Budgets and concurrency

Generation concurrency limits outstanding model requests. Evaluation concurrency
limits process workers. They are independent settings: increasing workers does
not increase model-request concurrency. Model-call timeouts and per-episode
execution deadlines are also separate.

Budget meanings depend on the optimizer: EliteSearch uses population and
generation counts; AlphaEvolve uses proposal counts; LineageSearch uses trial
attempts. Repair, reflection, and prompt evolution may add model requests.
Candidate evaluations also multiply by the number of seeds, while cached
successes can reduce actual executions. Compare recorded calls, episodes, and
costs rather than generation counts alone.

The unified CLI offers call, token, and spend caps. Its budget provider reserves
full configured input/output allowances before each call, including retries;
failed or cancelled calls still consume reservations. A spend cap requires
caller-supplied input/output price ceilings per million tokens. Omitted caps
are unlimited. Reported actual usage can be unknown when the provider omits it.
See [CLI budgets](../api/cli.md#generation-and-budget-options).

## Selection and recovery

Optimize on search seeds, select finalists on validation seeds, and measure the
chosen winner on held-out test seeds. The CLI requires these panels to be
disjoint. It ranks finalists on validation means when provided, otherwise takes
the optimizer's first ranked candidate. Testing does not reselect the winner.

`search(..., on_checkpoint=callback)` invokes a synchronous callback after
proposal and update, and attempts a checkpoint before propagating an error or
cancellation. Saving a `Run` alone does not serialize arbitrary optimizer state.
See [runs and recovery](runs.md) for the actual supported resume paths and the
[optimizer reference](../api/optimizers.md) for algorithm scope.
