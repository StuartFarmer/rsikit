# Circle packing

`CirclePackingEnv` defines a one-step Gymnasium task. Its observation contains
unit-square bounds and circle count; its action is a float64 `(count, 3)` array of
`(x, y, radius)` rows. Positive circles must fit without overlap (tolerance `1e-9`).
Feasible packings earn their total radius; infeasible geometry earns zero.

From the repository root:

```sh
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
.venv/bin/python -B -m rsikit.examples.circle_packing
```

The supplied `initial.py` exports a `Solution(Policy)` placing ten radius-0.1
circles on a grid, scoring 1.0. Supply another Python program as the CLI's optional
positional argument. It must export the same policy interface.

The environment owns canonical task instructions. The runner passes those to the
policy and records them with the episode. Policy source executes only inside the
restricted worker; geometry validation stays in the trusted environment.

The previous generation/repair/plotting CLI has been retired. See the
[inner-loop API](../../../docs/INNER_LOOP.md).
