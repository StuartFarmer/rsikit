"""Run one Paper 1 experiment; save everything for offline notebook analysis."""

import argparse
import asyncio
import csv
import inspect
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distributions
from pathlib import Path
from statistics import fmean
from uuid import uuid4

import gymnasium as gym
from slick import prompts
from slick.providers import OpenRouterAPI
from sqlalchemy import inspect as inspect_database
from sqlmodel import select

from examples.elitesearch import measure, run_search
from research import elitesearch
from research.elitesearch import Config, EliteSearch, Generation, Organism
from research.rollouts import Rollouts
from rsikit import Executor, Run
from rsikit.policy import Policy

TASKS = (
    "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
    "Pendulum-v1",
    "Blackjack-v1",
    "FrozenLake-v1",
    "CliffWalking-v1",
    "Taxi-v4",
    "LunarLander-v3",
    "BipedalWalker-v3",
    "CarRacing-v3",
    "LunarLanderContinuous-v3",
    "BipedalWalkerHardcore-v3",
    "FrozenLake8x8-v1",
    "CliffWalkingSlippery-v1",
)
ROOT = Path(__file__).resolve().parents[2]
EXECUTION_OPTIONS = (
    "episode_timeout",
    "concurrency",
    "generation_concurrency",
    "generations",
)
RANDOM_SOURCE = """from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return self.action_space.sample()
"""


def make_environment(name, *, max_steps=None):
    """Use installed Gymnasium defaults and its own observation/action documentation."""
    if name not in TASKS:
        raise ValueError(f"Unsupported environment: {name}")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    limit = max_steps or gym.spec(name).max_episode_steps or 500
    env = gym.Wrapper(gym.make(name, max_episode_steps=limit))
    # A wrapper retains instructions when Box2D's EzPickle recreates the base env.
    settings = {
        key: parameter.default
        for key, parameter in inspect.signature(type(env.unwrapped).__init__).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    settings.update(env.spec.kwargs)
    env.paper_settings = json.loads(
        json.dumps(dict(id=name, kwargs=settings, max_episode_steps=limit))
    )
    env.instructions = (
        f"Environment: {name}. Maximize cumulative episode reward.\n"
        f"Resolved settings (override documentation examples): {json.dumps(env.paper_settings)}\n"
        f"Observation space: {env.observation_space}\nAction space: {env.action_space}\n"
        f"Episode limit: {limit} steps. You receive observations only, not info/action masks.\n"
        f"Installed Gymnasium environment documentation:\n{inspect.getdoc(type(env.unwrapped))}"
    )
    if name.startswith("BipedalWalker"):
        env.instructions += """

Implementation details for both BipedalWalker variants (override vague documentation):
The float32 observation has 24 entries, with angles in radians:
[0] hull angle; [1] hull angular velocity * 2 / 50;
[2] horizontal velocity * 0.12; [3] vertical velocity * 0.08;
[4] first hip angle; [5] first hip angular velocity / 4;
[6] first knee angle + 1; [7] first knee angular velocity / 6;
[8] first lower-leg ground contact (0 or 1);
[9] second hip angle; [10] second hip angular velocity / 4;
[11] second knee angle + 1; [12] second knee angular velocity / 6;
[13] second lower-leg ground contact (0 or 1);
[14:24] ten lidar hit fractions in [0, 1], with 1 meaning no hit within range.
Ray i=0..9 points from the hull along (sin(0.15*i), -cos(0.15*i)),
with range 160/30 world units. Absolute position is not observed.
Action is a float32 vector of shape (4,) in [-1, 1], ordered
[first hip, first knee, second hip, second knee]. Each sign sets motor direction;
absolute magnitude sets the torque limit to 80*abs(action), rather than scaling
motor speed continuously. Hip speed magnitude is 4, knee speed magnitude is 6.
Simulation timestep is 1/50 second. Reward is the change in
130*x/30 - 5*abs(hull angle), minus 0.028*sum(abs(action)).
Hull-ground contact or x<0 ends the episode with reward -100 for that step;
reaching the terrain's right end also terminates. The step cap truncates episodes.
"""
    return env


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


class LoggedOpenRouter(OpenRouterAPI):
    """Record usage before Slick discards the native response metadata."""

    def __init__(self, *, output, **kwargs):
        super().__init__(**kwargs)
        self.output = output

    async def _asend(self, request):
        started = time.monotonic()
        record = dict(started_at=datetime.now(timezone.utc).isoformat(), request=request)
        try:
            response = await super()._asend(request)
            raw = response.model_dump(mode="json")
            record.update({key: raw.get(key) for key in ("id", "model", "provider", "usage")})
            return response
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            # No await in this append: concurrent calls cannot interleave JSON lines.
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, default=str) + "\n")


def snapshot(path):
    files = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "Dockerfile",
        ROOT / "scripts/run",
        ROOT / ".dockerignore",
        ROOT / "research/__init__.py",
        ROOT / "research/rollouts.py",
        ROOT / "research/rewards.py",
        ROOT / "examples/elitesearch.py",
    ]
    for package in ("rsikit", "research/elitesearch"):
        files.extend(
            file for file in (ROOT / package).rglob("*") if file.suffix in (".py", ".j2", ".txt")
        )
    files.extend(
        file for file in Path(__file__).parent.iterdir() if file.suffix in (".py", ".md", ".ipynb")
    )
    with zipfile.ZipFile(path / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(ROOT))


async def export_curves(agent, run, rollouts, heldout_seeds):
    """Evaluate fixed historical winners after search; never select on test scores."""
    organisms = {row.id: row for row in agent.organisms}
    policies = {policy.id: policy for policy in run.policies()}
    winners = {
        organisms[g.elite_ids[0]].policy_id
        for g in agent.generations
        if g.status == "completed" and g.elite_ids
    }
    random_policy = Policy.from_text(RANDOM_SOURCE, name="Random action reference")
    measurements = await measure(
        rollouts, [policies[key] for key in sorted(winners)] + [random_policy], heldout_seeds
    )
    evidence = {
        key: dict(scores=result.scores, failure=result.failure)
        for key, result in measurements.items()
    }
    evidence["random"] = evidence.pop(random_policy.id)
    write_json(run.path / "heldout.json", evidence)
    rows = []
    for generation in agent.generations:
        if generation.status != "completed":
            continue
        row = dict(
            generation=generation.number,
            policy_id=None,
            search_mean=None,
            heldout_mean=None,
            heldout_n=0,
            heldout_error="No valid elite",
            attempts=0,
            calls=0,
            repairs=0,
            discarded=0,
        )
        attempts = [o for o in agent.organisms if o.generation <= generation.number]
        row.update(
            attempts=len(attempts),
            calls=sum(len(o.calls) for o in attempts),
            repairs=sum(o.repairs for o in attempts),
            discarded=sum(o.status == "discarded" for o in attempts),
        )
        if generation.elite_ids:
            winner = organisms[generation.elite_ids[0]]
            result = measurements[winner.policy_id]
            row.update(
                policy_id=winner.policy_id,
                search_mean=winner.score,
                heldout_mean=fmean(result.scores.values()) if result.scores else None,
                heldout_n=len(result.scores),
                heldout_error=result.failure or "",
            )
        rows.append(row)
    if rows:
        temporary = run.path / "curves.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(run.path / "curves.csv")


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--env", choices=TASKS, default="CartPole-v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue an existing Paper 1 run directory")
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--elites", type=int, default=5)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument(
        "--target-score",
        type=float,
        help="Stop when the best search mean reaches this score; defaults to Gymnasium's threshold",
    )
    parser.add_argument("--no-early-stop", action="store_true", help="Always run all generations")
    parser.add_argument("--max-repairs", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--generation-concurrency", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent episode processes")
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=10,
        help="Wall-clock seconds per episode evaluation (default: 10)",
    )
    parser.add_argument("--max-steps", type=int, help="Override the environment episode limit")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=list(range(100, 200)))
    args = parser.parse_args(argv)
    if args.list_envs:
        print("\n".join(TASKS))
        return
    saved = None
    history = []
    if args.resume is not None:
        try:
            saved = json.loads((args.resume / "experiment.json").read_text())
            history_path = args.resume / "attempts.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            if not isinstance(saved, dict) or not isinstance(history, list):
                raise ValueError("Invalid experiment/attempt metadata")
            required = (
                "config",
                "env",
                "seeds",
                "heldout_seeds",
                "instructions",
                "environment",
                "versions",
            )
            if any(key not in saved for key in required):
                raise ValueError("Not a Paper 1 experiment manifest")
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot resume: {exc}")
        inherited = {key: saved.get(key, parser.get_default(key)) for key in vars(args)}
        for key in EXECUTION_OPTIONS:
            if history and key in history[-1]:
                inherited[key] = history[-1][key]
        inherited.update(output=args.resume, resume=args.resume)
        parser.set_defaults(**inherited)
        args = parser.parse_args(argv)
        if args.output.resolve() != args.resume.resolve():
            parser.error("--output cannot change the resumed run directory")
        for key, previous in inherited.items():
            if (
                key not in (*EXECUTION_OPTIONS, "output", "resume")
                and getattr(args, key) != previous
            ):
                parser.error(f"Cannot change --{key.replace('_', '-')} when resuming")
        if args.generations < inherited["generations"]:
            parser.error("--generations is a total budget and cannot decrease when resuming")
    if args.output is None:
        parser.error("--output is required; choose a new directory for each search")
    for name in (
        "population",
        "elites",
        "generations",
        "max_output_tokens",
        "generation_concurrency",
        "concurrency",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.max_repairs < 0 or args.search_seed < 0:
        parser.error("--max-repairs and --search-seed must be nonnegative")
    if not math.isfinite(args.episode_timeout) or args.episode_timeout <= 0:
        parser.error("--episode-timeout must be positive and finite")
    if args.target_score is not None and not math.isfinite(args.target_score):
        parser.error("--target-score must be finite")
    if any(seed < 0 for seed in args.seeds + args.heldout_seeds):
        parser.error("Episode seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.heldout_seeds)) != len(
        args.heldout_seeds
    ):
        parser.error("Duplicate episode seeds are not independent episodes")
    if set(args.seeds) & set(args.heldout_seeds):
        parser.error("Search and held-out seeds must be disjoint")
    if args.output.exists() and args.resume is None:
        parser.error(f"Output already exists: {args.output}")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running")
    config = Config(
        population_size=args.population,
        elite_size=args.elites,
        generations=args.generations,
        max_repairs=args.max_repairs,
        generation_concurrency=args.generation_concurrency,
        target_score=(
            None
            if args.no_early_stop
            else args.target_score
            if args.target_score is not None
            else gym.spec(args.env).reward_threshold
        ),
    )
    if saved is not None:
        config = replace(
            Config(**saved["config"]),
            generations=args.generations,
            generation_concurrency=args.generation_concurrency,
        )
    started = time.monotonic()
    previous_templates = prompts.TEMPLATE_ROOT
    with make_environment(args.env, max_steps=args.max_steps) as env:
        if saved is not None:
            if (
                saved["environment"] != env.paper_settings
                or saved["versions"].get("gymnasium") != gym.__version__
            ):
                parser.error("The installed environment differs from the saved run")
            env.instructions = saved.get("environment_instructions", saved["instructions"])
        environment_instructions = env.instructions
        stopping = (
            "Early stopping is disabled; maximize mean search reward throughout the generation budget."
            if config.target_score is None
            else f"Stop after a completed generation when the best elite has mean search reward >= {config.target_score:g}."
        )
        env.instructions += (
            "\n\nEvaluation contract (takes precedence over documentation examples):\n"
            "Maximize the arithmetic mean of undiscounted cumulative episode rewards across "
            "the fixed search seed panel. Higher is better, including when all rewards are negative.\n"
            f"{stopping}\n"
            "Held-out episodes are for reporting only and never guide selection or stopping.\n"
            f"Wall-clock evaluation limit: {args.episode_timeout:g} seconds per episode, per seed, "
            "for the worker to return its episode result. Policy initialization and environment "
            "steps must fit this limit. Docker startup, queueing and final process cleanup are "
            "outside this limit. This is not a total budget across all seeds.\n"
            "A timeout is an evaluation failure, not a truncated successful episode or a zero reward. "
            "Search-time failures enter the repair loop, within the repair budget.\n"
            "The policy receives observations, not rewards or info/action masks; episode state "
            "must be reset between episodes. Evaluation feedback reaches the LLM between proposals.\n"
        )
        executor = Executor(
            concurrency=args.concurrency,
            episode_timeout=args.episode_timeout,
        )
        try:
            run_context = (
                Run.open(args.resume)
                if args.resume
                else Run.create(name=f"paper1-{args.env}", path=args.output)
            )
        except BlockingIOError:
            parser.error("Run is locked by another process; stop that process before resuming")
        async with executor, run_context as run:
            rollouts = Rollouts(env, executor, run)
            provider = LoggedOpenRouter(
                output=run.path / "llm_calls.jsonl",
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                timeout=120,
            )

            async def evaluate(policies):
                return await measure(rollouts, policies, args.seeds)

            agent = EliteSearch(
                "Maximize cumulative episode reward in the described environment.",
                provider,
                evaluate,
                context=env.instructions,
                config=config,
                seed=args.search_seed,
            )
            if args.resume:
                with run.database() as db:
                    tables = set(inspect_database(db.connection()).get_table_names())
                    required = {"elitesearch_organism", "elitesearch_generation"}
                    if tables & required and not required <= tables:
                        raise ValueError("Incomplete search checkpoint tables")
                    if required <= tables:
                        agent.restore(
                            list(db.exec(select(Organism).order_by(Organism.id))),
                            list(db.exec(select(Generation).order_by(Generation.number))),
                        )
                    elif run.policies():
                        raise ValueError("Saved policies have no search checkpoint")
                if agent._seed_panel is not None and agent._seed_panel != set(args.seeds):
                    raise ValueError("Checkpoint seed panel differs from experiment.json")
                if not history:
                    old_status = run.path / "status.json"
                    history.append(
                        dict(
                            **(json.loads(old_status.read_text()) if old_status.exists() else {}),
                            **{
                                key: saved.get(key, getattr(args, key)) for key in EXECUTION_OPTIONS
                            },
                            kind="original",
                            source="source.zip",
                        )
                    )
                backup = run.path / "resumes" / f"{len(history):03}-{uuid4().hex[:8]}"
                backup.mkdir(parents=True)
                for name in (
                    "status.json",
                    "summary.json",
                    "curves.csv",
                    "heldout.json",
                    "leaderboard.json",
                    "best.py",
                ):
                    if (run.path / name).exists():
                        shutil.copy2(run.path / name, backup / name)
            else:
                backup = run.path
            state = dict(status="running", started_at=datetime.now(timezone.utc).isoformat())
            attempt = dict(
                **state,
                **{key: getattr(args, key) for key in EXECUTION_OPTIONS},
                kind="resume" if args.resume else "original",
                source=str((backup / "source.zip").relative_to(run.path)),
                context=str((backup / "context.txt").relative_to(run.path)),
                context_sha256=sha256(env.instructions.encode()).hexdigest(),
                config=asdict(config),
                checkpoint_generation=len(agent.generations),
                docker_image=os.environ.get("RSIKIT_IMAGE_ID"),
                git_revision=os.environ.get("RSIKIT_GIT_REVISION"),
            )
            history.append(attempt)
            write_json(run.path / "attempts.json", history)
            write_json(run.path / "status.json", state)
            print(f"Output: {run.path}", flush=True)
            print(
                f"Early-stop search target: {config.target_score if config.target_score is not None else 'disabled'}",
                flush=True,
            )
            try:
                (backup / "context.txt").write_text(env.instructions, encoding="utf-8")
                snapshot(backup)
                if saved is None:
                    write_json(
                        run.path / "experiment.json",
                        dict(
                            **vars(args),
                            reporting_target=(
                                args.target_score
                                if args.target_score is not None
                                else env.spec.reward_threshold
                            ),
                            config=asdict(config),
                            environment=env.paper_settings,
                            instructions=env.instructions,
                            environment_instructions=environment_instructions,
                            versions={
                                item.metadata["Name"]: item.version for item in distributions()
                            },
                            git_revision=os.environ.get("RSIKIT_GIT_REVISION"),
                            docker_image=os.environ.get("RSIKIT_IMAGE_ID"),
                        ),
                    )
                prompts.TEMPLATE_ROOT = Path(elitesearch.__file__).parent / "prompts"
                await run_search(
                    agent,
                    run,
                    rollouts,
                    seeds=args.seeds,
                    heldout_seeds=args.heldout_seeds,
                )
                await export_curves(agent, run, rollouts, args.heldout_seeds)
                state["status"] = "completed"
            except BaseException as exc:
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                prompts.TEMPLATE_ROOT = previous_templates
                state.update(
                    elapsed_seconds=time.monotonic() - started,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                attempt.update(state)
                write_json(run.path / "attempts.json", history)
                write_json(run.path / "status.json", state)
    print(f"Finished: {args.output.resolve() / 'curves.csv'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
