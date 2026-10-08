# Customization

The smallest extension is a custom policy. A complete custom system also chooses
how to propose candidates, collect evidence, and select a winner. The
[custom-system example](../examples/custom-system.md) implements that loop without
model calls.

## Custom policies

Implement `Policy.act`, clear your episode state in `reset`, and close any resources
you allocate. Policies receive observations and their observation/action spaces;
the interface does not pass the live environment. Validate actions against the
space's shape, dtype, and bounds. For saved/generated source, use the class name
`Solution` and include every import needed by that source.

## Custom Gymnasium environments

A Python integration needs only a Gymnasium environment that the executor can
serialize. This one-step environment illustrates the required reset and step
contracts:

```python
import gymnasium as gym

class ChooseOne(gym.Env):
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(2)
    instructions = "Return action 1 to earn reward 1; action 0 earns 0."

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(action)
        return 0, float(action == 1), True, False, {}
```

Use it as the `environment` passed to direct evaluation or `Run.evaluate`.
Use `self.np_random` for seeded environment randomness. For a fitness distinct from
cumulative reward, return a finite numeric `fitness` in the final info. Keep
observations, actions, and infos within the [episode value limits](policies.md).
Own external handles carefully: worker execution requires serializable inputs
and closes its private environment copy. A different environment configuration
belongs in a different `Run`.

## Custom optimizers

The Python `Optimizer` protocol is structural; subclassing is optional:

- `done` reports whether search has finished.
- `best` returns the best accepted definition, or `None`.
- `async propose()` returns unique policy definitions for one round.
- `update(results)` consumes all outstanding IDs as `{id: {seed: Episode}}`.

Reject invalid/incomplete feedback before changing optimizer state. Include failed
candidates in feedback and exclude their partial rewards from selection. Keep
successful comparisons on the same seed panel. Returning an empty proposal while
`done` remains false is an error. `search` owns the round ordering; your optimizer
owns its budget, selection rule, and checkpoint representation.

A deterministic proposal strategy can later call `generate` or a provider without
changing evaluation or persistence. See the complete
[offline optimizer implementation](../examples/custom-system.md).

## CLI adapters

The unified CLI loads a local optimizer file with `--optimizer path/to/adapter.py`.
This adapter is a separate contract from the Python `Optimizer` protocol. Export:

| Export | Contract |
| --- | --- |
| `Options` | A Pydantic model with `extra="forbid"`; `generation` and `videos` are reserved runtime fields. |
| `add_arguments(parser)` | Register flags whose destination names match `Options` fields. Defaults/requiredness are resolved after merging YAML and flags. |
| `async optimize(*, task, provider, evaluate, run, options, seed)` | Run the search and return policy definitions ranked best first. `evaluate(policies)` returns complete episode feedback. |

`options` contains validated adapter fields plus runtime `generation` and `videos`
sections. Use the supplied provider to retain shared call budgeting. Use the
supplied `Run` for records and the supplied evaluation callback for evidence.
The runner filters finalists to policies with successful search-panel scores.

A custom CLI environment file must export `environment`, usually a
`research.experiment.EnvironmentDefinition`. It supplies an `Options` model,
`add_arguments`, `describe(options, evaluation)`, and an async context manager
`open_evaluator(*, options, evaluation, run)` yielding
`evaluate(policies, seeds)`. Environment flags must start with `--env-`.
The definition also declares evaluation defaults, supported evaluation fields,
score keys, protocol name, and provenance callback. The bundled
`research.environments` integration is the reference implementation.

Loading either adapter imports and executes that Python file. YAML paths resolve
relative to the config file; CLI paths resolve relative to the working directory.
See [CLI configuration](../api/cli.md) for the shared schema.
