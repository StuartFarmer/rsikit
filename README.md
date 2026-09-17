# RSIKit

A small inner loop for class-based policies in
[Gymnasium](https://gymnasium.farama.org/) environments.

- `Policy`: task instructions, spaces, and async reset/act/close.
- `run_episode`: run trusted policy classes in existing Gymnasium environments.
- `run_program`: evaluate generated classes inside Docker.

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

Runtime dependencies are Gymnasium and NumPy. No sibling checkout is required.

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

## Checks

```sh
.venv/bin/python -B -m unittest discover -s rsikit/tests -v
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Docker checks skip explicitly when Docker or the worker image is unavailable.
The repository contains only the inner loop, circle-packing and CartPole examples,
and their checks. Earlier research and optimizer implementations remain in Git
history and on `codex/gymnasium-inner-loop`.
