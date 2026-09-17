"""Evaluate policy source through RSIKit's isolated Gymnasium inner loop."""

import json
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean

import gymnasium as gym

from rsikit.episode import PolicyError
from rsikit.sandbox import run_program

from .agent import Evaluation


async def evaluate_program(
    source: str,
    make_env: str | Callable[[], gym.Env],
    *,
    seeds: Sequence[int] = (0,),
    max_steps: int | None = None,
    instructions: str | None = None,
    call_timeout: float = 10.0,
) -> Evaluation:
    """Maximize mean episode return over fixed seeds, running source only in Docker.

    Policy failures reject the candidate. Infrastructure/environment errors abort
    the search. Custom objectives, cascades, and diversity cells use an injected
    evaluator returning Evaluation instead.
    """
    episodes = []
    with tempfile.TemporaryDirectory(prefix="rsikit-eval-") as directory:
        program = Path(directory) / "solution.py"
        program.write_text(source, encoding="utf-8")
        for seed in seeds:
            try:
                _, _, terminated, truncated, info = await run_program(
                    program,
                    make_env,
                    env_seed=seed,
                    policy_seed=seed,
                    max_steps=max_steps,
                    instructions=instructions,
                    call_timeout=call_timeout,
                )
            except PolicyError as exc:
                return Evaluation(
                    error=f"Seed {seed}: {type(exc).__name__}: {exc}",
                    feedback=json.dumps(episodes, default=str),
                )
            episodes.append(
                {
                    "seed": seed,
                    "reward": info["episode"]["r"],
                    "length": info["episode"]["l"],
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": {key: value for key, value in info.items() if key != "episode"},
                }
            )
    return Evaluation(
        metrics={"reward": fmean(episode["reward"] for episode in episodes)},
        feedback=json.dumps(episodes, default=str),
    )
