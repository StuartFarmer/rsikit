# RSIKit

Evolve and evaluate class-based policies in
[Gymnasium](https://gymnasium.farama.org/) environments.

- `Policy`: task instructions, spaces, and async reset/act/close.
- `run_episode`: run trusted policy classes in existing Gymnasium environments.
- `run_program`: evaluate generated classes inside Docker.
- `AlphaEvolve`: evolve programs using Slick, Pydantic, and Gymnasium feedback.

```python
from rsikit import Policy, run_episode


class RandomPolicy(Policy):
    async def act(self, observation):
        return self.action_space.sample()


# Inside an async function:
observation, reward, terminated, truncated, info = await run_episode("CartPole-v1", RandomPolicy)
print(info["episode"])  # Gymnasium totals: r (return), l (length), t (seconds)
```

Use an environment ID or a factory such as `lambda: gym.make("FrozenLake-v1")`.
Instructions and seeds are optional. Gymnasium supplies the spaces, time limits,
and episode statistics; there are no custom episode or transition objects.

## Setup

```sh
uv venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -B -m rsikit.examples.cartpole
```

Runtime dependencies are Gymnasium, NumPy, Slick (`slick-ai`), and Pydantic.
No sibling checkout or `slick-bits` dependency is required.

## Isolated programs

```sh
docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile .
.venv/bin/python -B -m rsikit.examples.circle_packing
.venv/bin/python -B -m rsikit.examples.circle_packing /path/to/solution.py
```

A program exports `Solution(Policy)`. One policy instance persists throughout an
episode. The environment and scoring stay outside the restricted worker.

See the [API guide](docs/INNER_LOOP.md) for task instructions, class contracts,
results, and execution limits.

## AlphaEvolve

See the [AlphaEvolve guide](docs/ALPHAEVOLVE.md) for the Python API, search controls,
and structured generation contracts. Run the CartPole example with an API model:

```sh
uv pip install --python .venv/bin/python -e '.[openai]'
# Set OPENAI_API_KEY in your environment first; this command makes paid model calls.
.venv/bin/python -B -m rsikit.examples.alphaevolve --model YOUR_MODEL --attempts 10
```

Build the Docker worker above before running the search. The best evaluated
`Solution` is written to `runs/cartpole/solution.py`.

## Checks

```sh
.venv/bin/python -B -m unittest discover -s rsikit/tests -v
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Docker checks skip explicitly when Docker or the worker image is unavailable.
The core remains a Gymnasium episode runner. AlphaEvolve is a separate module
that generates policies and evaluates them through that runner.
