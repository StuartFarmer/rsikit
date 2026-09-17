# RSIKit

Class-based policies evaluated in [Gymnasium](https://gymnasium.farama.org/) environments.
Each task defines a standard environment; a `Solution(Policy)` class chooses actions;
one episode runner records rewards, behavior, and failures. No weight training.

```sh
uv venv
uv pip install --python .venv/bin/python -e ../slick -e '.[dev]'
.venv/bin/python -B -m rsikit.examples.cartpole
```

The setup uses the adjacent Slick checkout for the research modules. The active
inner loop uses Gymnasium and NumPy; it makes no model API calls.

## Run a generated class

Build the isolated worker from the repository root:

```sh
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
.venv/bin/python -B -m rsikit.examples.circle_packing
.venv/bin/python -B -m rsikit.examples.circle_packing /path/to/solution.py
```

A program exports `Solution(Policy)` with async `act(observation)`. The inherited
constructor receives observation/action spaces and `instructions=...`; async
`reset(seed=...)` initializes episode state, and async `close()` releases resources.
Generated code executes in Docker, while the task and reward computation stay
outside it. One policy instance persists across an episode's actions.

See the [API guide](docs/INNER_LOOP.md) for trusted local classes, task instructions,
seeds, episode records, execution limits, and supported isolated spaces.

## Check the new API

```sh
.venv/bin/python -B -m unittest rsikit.tests.test_inner_loop rsikit.tests.test_sandbox_episode -v
```

Docker checks explicitly skip when Docker or the image is unavailable. They never
execute generated source on the host as a fallback. These smoke checks cover the
new API; comprehensive hardening is separate work.

Version 0.2 intentionally replaces the old root optimizer/evaluator API. Selection,
archives, islands, proposers, and other outer-loop modules remain research material
awaiting integration. Historical examples/tests using the removed API are not the
active validation suite. No backwards-compatibility layer is supplied.

[Implementation plan](docs/superpowers/plans/2026-09-17-gymnasium-inner-loop.md) ·
[Design](docs/superpowers/specs/2026-09-17-gymnasium-inner-loop-design.md) ·
[Module research](outputs/rsikit-conceptual-modules-comparison.md)

Attribution remains in [NOTICE](rsikit/NOTICE) and
[LICENSE.funsearch](rsikit/LICENSE.funsearch).
