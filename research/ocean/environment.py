"""Use upstream Ocean factories and bindings with the pinned episodic compatibility patch."""

import hashlib
import importlib
import importlib.util
import json
from argparse import BooleanOptionalAction
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from pydantic import Field, model_validator

from research.experiment import EnvironmentDefinition, Options

UPSTREAM = "3b5c6046bb8b46685d62d151720025507e3418c2"
PROTOCOL = "ocean-upstream-episodic-v2"


def factory(name="g2048"):
    if name == "maze":
        from .maze import Maze

        return Maze
    try:
        from pufferlib.ocean import env_creator
    except ImportError as exc:
        raise ImportError("Install Ocean first: python scripts/install_ocean.py") from exc
    constructor = env_creator(f"puffer_{name}")
    if not callable(constructor):
        raise ValueError(f"No installed Ocean environment named {name!r}")
    return constructor


def metadata(name="g2048"):
    if name == "maze":
        from .maze import metadata as maze_metadata

        return maze_metadata()
    try:
        import pufferlib
    except ImportError as exc:
        raise ImportError("Install Ocean first: python scripts/install_ocean.py") from exc

    provenance = Path(pufferlib.__file__).with_name("rsikit_ocean.json")
    installed = json.loads(provenance.read_text()) if provenance.exists() else {}
    if installed.get("upstream") != UPSTREAM or installed.get("episodic_patch") != "episodes-v1":
        raise RuntimeError("Ocean needs the episode fix; run python scripts/install_ocean.py")
    binding = importlib.util.find_spec(f"pufferlib.ocean.{name}.binding")
    if binding is None:
        raise ImportError(
            f"Build the upstream binding: python scripts/install_ocean.py --envs {name}"
        )
    library = Path(binding.origin)
    native = importlib.import_module(f"pufferlib.ocean.{name}.binding")
    if getattr(native, "rsikit_episodic_patch", None) != "episodes-v1":
        raise RuntimeError(
            f"Rebuild the episode fix: python scripts/install_ocean.py --envs {name}"
        )
    return dict(
        protocol=PROTOCOL,
        upstream=UPSTREAM,
        episodic_patch=installed["episodic_patch"],
        environment=name,
        native_library=str(library),
        native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        seed_semantics="one episode per seed; independent per-lane rand_r state; batch width only limits concurrency",
    )


class G2048Options(Options):
    reward_scaler: float = Field(default=1, gt=0)
    can_go_over_65536: bool = False
    endgame_env_prob: float = Field(default=0, ge=0, le=1)
    scaffolding_ratio: float = Field(default=0, ge=0, le=1)
    use_heuristic_rewards: bool = False
    snake_reward_weight: float = 0
    use_sparse_reward: bool = False


class BreakoutOptions(Options):
    frameskip: int = Field(default=4, ge=1, le=2147483647)
    width: int = Field(default=576, ge=1, le=2147483647)
    height: int = Field(default=330, ge=1, le=2147483647)
    paddle_width: int = Field(default=62, ge=1)
    paddle_height: int = Field(default=8, ge=1)
    ball_width: int = Field(default=32, ge=1)
    ball_height: int = Field(default=32, ge=1)
    brick_width: int = Field(default=32, ge=1)
    brick_height: int = Field(default=12, ge=1)
    brick_rows: int = Field(default=6, ge=1)
    brick_cols: int = Field(default=18, ge=1)
    initial_ball_speed: float = Field(default=256, gt=0)
    max_ball_speed: float = Field(default=448, gt=0)
    paddle_speed: float = Field(default=620, gt=0)
    continuous: bool = False

    @model_validator(mode="after")
    def valid_geometry(self):
        if (
            self.paddle_width >= self.width
            or self.ball_width >= self.width
            or self.paddle_height + self.ball_height >= self.height
            or self.brick_cols * self.brick_width > self.width
            or self.brick_rows * self.brick_height >= self.height
        ):
            raise ValueError("Breakout objects and bricks must fit within the arena")
        if self.initial_ball_speed > self.max_ball_speed:
            raise ValueError("initial_ball_speed must not exceed max_ball_speed")
        return self


class MazeOptions(Options):
    # Fixed dimensions make map identity and the episode horizon explicit.
    map_size: int = Field(default=15, ge=5, le=47)

    @model_validator(mode="after")
    def odd_size(self):
        if self.map_size % 2 == 0:
            raise ValueError("Maze map_size must be odd")
        return self


def arguments(parser, *, name):
    if name == "maze":
        parser.add_argument("--env-map-size", dest="map_size", type=int)
    elif name == "g2048":
        for key in (
            "reward_scaler",
            "endgame_env_prob",
            "scaffolding_ratio",
            "snake_reward_weight",
        ):
            parser.add_argument("--env-" + key.replace("_", "-"), dest=key, type=float)
        for key in ("can_go_over_65536", "use_heuristic_rewards", "use_sparse_reward"):
            parser.add_argument(
                "--env-" + key.replace("_", "-"), dest=key, action=BooleanOptionalAction
            )
    else:
        for key in (
            "frameskip",
            "width",
            "height",
            "paddle_width",
            "paddle_height",
            "ball_width",
            "ball_height",
            "brick_width",
            "brick_height",
            "brick_rows",
            "brick_cols",
        ):
            parser.add_argument("--env-" + key.replace("_", "-"), dest=key, type=int)
        for key in ("initial_ball_speed", "max_ball_speed", "paddle_speed"):
            parser.add_argument("--env-" + key.replace("_", "-"), dest=key, type=float)
        parser.add_argument("--env-continuous", dest="continuous", action=BooleanOptionalAction)


def definition(name):
    if name not in ("g2048", "breakout", "maze"):
        raise ValueError(f"Unsupported Ocean environment {name!r}; choose g2048, breakout or maze")
    return EnvironmentDefinition(
        Options={"g2048": G2048Options, "breakout": BreakoutOptions, "maze": MazeOptions}[name],
        add_arguments=partial(arguments, name=name),
        describe=partial(describe, name),
        open_evaluator=partial(open_evaluator, name),
        evaluation_defaults=dict(
            batch_size=32, max_steps=2000, score_key="merge_score" if name == "g2048" else "return"
        ),
        supported_evaluation_fields=frozenset({"batch_size", "max_steps", "score_key"}),
        score_keys=("return", "score", "episode_return", "perf")
        + (("merge_score",) if name == "g2048" else ()),
        protocol=PROTOCOL,
        provenance=partial(metadata, name),
    )


CONTEXT = """Upstream Ocean g2048: deterministic Python/NumPy batch policy.
Observation is uint8 (B,289). Columns 0:16 are encoded tile magnitudes;
16:32 are empty flags; 32:288 reshape to (B,16,16) one-hot tile exponents
1 through 16; column 288 is a snake-pattern flag. Decode a cell as zero if
its empty flag is set, otherwise argmax(one_hot) + 1.
Return integer actions (B,): 0=up, 1=down, 2=left, 3=right.
Policies must be stateless and deterministic, with independent row decisions.
B varies between batches. Finished rows freeze until all rows finish; their actions are ignored.
Each seed runs one episode, stopping at termination or the evaluator step cap.
No environment, file, process, network, clock or evaluator access.
"""
BREAKOUT_CONTEXT = """Upstream Ocean Breakout: Python/NumPy batch policy.
Observation is float32 (B,10 + brick_rows * brick_cols): paddle x/y, ball x/y,
ball vx/vy, balls fired, score, remaining balls, paddle width, then brick states.
Positions (top-left coordinates) are normalized by arena dimensions, velocities by 512.
Balls fired and remaining balls are divided by 5, score by 864, paddle width by 62.
Return integer actions (B,): 0=noop, 1=left, 2=right.
Policies must be stateless and deterministic, with independent row decisions.
B varies between batches. Finished rows freeze and their actions are ignored.
Each seed runs one episode, stopping at termination or the evaluator step cap.
No environment, file, process, network, clock or evaluator access.
"""
MAZE_CONTEXT = """Ocean Maze: deterministic Python/NumPy batch policy with episode memory.
Observation is uint8 (B,121), reshaped (B,11,11), centered on the agent at [5,5].
Cells: 0=empty, 1=wall, 2=agent, 4=goal. Return integer actions (B,):
0=noop, 1=east, 2=north, 3=west, 4=south.
Each seed generates one 15x15 map; goal reward is 1, otherwise 0.
The native timeout is twice the map area (450 steps for 15x15), or the evaluator cap.
You may keep bounded per-row episode memory. Clear it in async reset(self, *, seed=None).
Rows remain stable within each batch; finished rows freeze and their actions are ignored.
A fresh policy instance starts each batch. Never share memory between rows.
No seeds, map bank, simulator, file, process, network, clock or evaluator access.
"""


def describe(name, options, evaluation):
    context = {"g2048": CONTEXT, "breakout": BREAKOUT_CONTEXT, "maze": MAZE_CONTEXT}[name]
    if name == "maze":
        context = context.replace("15x15", f"{options.map_size}x{options.map_size}").replace(
            "450 steps", f"{2 * options.map_size**2} steps"
        )
    if name == "breakout" and options.continuous:
        context = context.replace(
            "Return integer actions (B,): 0=noop, 1=left, 2=right.",
            "Return float32 actions (B,1), each in [-1,1], controlling paddle velocity.",
        )
    return (
        context + f"\nOptions: {options.model_dump_json()}\n"
        f"Score: {evaluation.score_key}; one episode per seed, at most "
        f"{evaluation.batch_size} concurrent games and {evaluation.max_steps} steps per episode. "
        "Return sums episode rewards; other scores read the final state, including capped episodes."
    )


@asynccontextmanager
async def open_evaluator(name, *, options, evaluation, run):
    from .evaluator import PanelEvaluator

    async with PanelEvaluator(
        run.path / "panels",
        workers=evaluation.workers,
        timeout=evaluation.timeout_per_seed,
        batch_size=evaluation.batch_size,
        max_steps=evaluation.max_steps,
        env_name=name,
        score_key=evaluation.score_key,
        env_kwargs=options.model_dump(),
    ) as evaluator:
        import asyncio

        from research.experiment import record_execution
        from research.ocean.evaluator import episode_panel

        async def evaluate(policies, seeds):
            async def measure(policy):
                from uuid import uuid4

                job_id = uuid4().hex
                event = None
                try:
                    event = await evaluator.submit(policy, seeds, job_id=job_id, record=True)
                    episodes = episode_panel(event, seeds)
                    for seed, episode in episodes.items():
                        run.save_episode(policy, seed, episode)
                    return policy.id, episodes
                finally:
                    if event is None:
                        event = next((e for e in evaluator.events if e["job_id"] == job_id), None)
                    if event is not None and event["started"] is not None:
                        record_execution(
                            policy.id,
                            seed_runs=len(seeds) if event["status"] == "ok" else None,
                            steps=event["steps"] if event["status"] == "ok" else None,
                        )

            return dict(await asyncio.gather(*(measure(p) for p in policies)))

        yield evaluate
