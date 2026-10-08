# Environments

Use ordinary Gymnasium environments with `evaluate` and `Executor`. rsikit's
additional environments follow the same `reset` / `step` interface. Configure
one environment per saved Run; changing configuration invalidates episode reuse.
See [policies and evaluation](../guide/policies.md) and
[customization](../guide/customization.md).

These are simulation tasks. Bitcoin and PriceSeries replay historical rows;
changing the seed does not create a new market path. Use separate time intervals
and separate runs for held-out evaluation. Bring CSV data you have rights to use;
the bundled Bitcoin dataset retains CC BY-NC 4.0 terms.

<!-- api: rsikit.envs.BlackjackEnv
{members: []}
-->

<!-- api: rsikit.envs.CirclePackingEnv
{members: []}
-->

<!-- api: rsikit.envs.PriceSeriesEnv
{members: []}
-->

<!-- api: rsikit.envs.BitcoinEnv
{members: []}
-->

## Task presets

`rsikit.envs.tasks.make_environment(name, *, max_steps=None, render_mode=None,
shoes_per_episode=24)` supplies task instructions alongside these presets:
`CartPole-v1`, `LunarLander-v3`, `BipedalWalker-v3`, `Blackjack`, and `Bitcoin`.
LunarLander uses continuous actions and wind. Box2D tasks require the `box2d`
extra. Renderers require the `video` extra. The [CLI](cli.md) additionally exposes
PriceSeries and optional Ocean environments through its own adapters.
