"""Generate, evaluate, and update Gymnasium policies through OpenRouter."""

import argparse
import asyncio
import json
import logging
import math
import os
import sqlite3
from contextlib import AsyncExitStack, closing
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, pstdev

import gymnasium as gym
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from slick import prompts
from slick.providers import OpenRouterAPI
from sqlalchemy import inspect
from sqlmodel import func, select

import alphaevolve
from alphaevolve import improved, original, paper
from alphaevolve.edits import check_program
from alphaevolve.history import Generation, history_records
from rsikit import Executor, Run
from rsikit.envs import BitcoinEnv, BlackjackEnv
from rsikit.episode import PolicyError
from rsikit.policy import _policy_class

TASKS = {
    "Bitcoin": "Maximize final USD wealth after BTC trading fees over the training period.",
    "Blackjack": "Maximize net profit over a finite blackjack shoe, with betting and sitting out.",
    "CartPole-v1": (
        "Balance the pole for as many steps as possible. Observation: [cart position, "
        "cart velocity, pole angle, pole angular velocity]. Action 0 pushes left; "
        "1 pushes right. Each surviving step earns 1 reward. The episode terminates "
        "when the pole tilts too far or the cart leaves the allowed range."
    ),
    "LunarLander-v3": (
        "Land safely near the pad at (0, 0), maximizing cumulative reward. Wind and "
        "turbulence are enabled. Observation indices: 0 horizontal position, 1 vertical "
        "position, 2 horizontal velocity, 3 vertical velocity, 4 angle, 5 angular "
        "velocity, 6 left leg contact, 7 right leg contact. Positions and velocities "
        "are normalized simulator values. Return a float32 array [main, lateral], "
        "each in [-1, 1]. Main < 0 turns the main engine off; main in [0, 1] maps to "
        "50-100% power. Lateral in (-0.5, 0.5) turns side engines off; below -0.5 "
        "fires the left booster, above 0.5 fires the right, with magnitude mapping "
        "to 50-100% power. Reward favors approaching the pad, slowing down, staying "
        "upright and leg contact; firing engines costs reward. A crash costs 100; "
        "a safe landing earns 100. Termination: crash, leaving the horizontal bounds, "
        "or coming to rest. Initial force and wind vary with the episode seed."
    ),
    "BipedalWalker-v3": (
        "Walk to the right over uneven terrain without falling, maximizing cumulative "
        "reward. Normal terrain, not hardcore. Observation indices: 0 hull angle, "
        "1 scaled hull angular velocity, 2-3 scaled horizontal/vertical velocity; "
        "4 hip angle, 5 scaled hip speed, 6 knee angle plus 1, 7 scaled knee speed, "
        "8 foot contact for the first leg; 9-13 the same five values for the second "
        "leg; 14-23 ten lidar fractions (0 near, 1 far). Return a float32 array of "
        "four motor commands in [-1, 1]: first hip, first knee, second hip, second "
        "knee. Sign sets motor direction; magnitude limits torque. Reward favors "
        "forward progress and an upright hull, with a motor-effort penalty. Falling "
        "costs 100. Termination: hull contacts ground or reaching the terrain end. "
        "Terrain and initial push vary with the episode seed."
    ),
}

FEATURE_BOUNDS = {
    "Bitcoin": {"mean_reward": (-10000, 1000000, 50), "reward_std": (0, 1, 1)},
    "Blackjack": {"mean_reward": (-1000, 1000, 40), "reward_std": (0, 500, 20)},
    "CartPole-v1": {"mean_reward": (0, 500, 20), "reward_std": (0, 250, 10)},
    "LunarLander-v3": {"mean_reward": (-500, 350, 34), "reward_std": (0, 500, 10)},
    "BipedalWalker-v3": {"mean_reward": (-200, 350, 22), "reward_std": (0, 200, 10)},
}


def _check_gym_evaluation(
    features, seeds, screening_seeds, screening_min_reward, objective="reward"
):
    if not seeds:
        raise ValueError("Evaluation requires at least one seed")
    if bool(screening_seeds) != (screening_min_reward is not None):
        raise ValueError("screening_seeds and screening_min_reward must be supplied together")
    if screening_min_reward is not None and not math.isfinite(screening_min_reward):
        raise ValueError("screening_min_reward must be finite")
    unknown = set(features) - {"mean_reward", "reward_std"}
    if unknown:
        raise ValueError(f"Unsupported Gym descriptors: {', '.join(sorted(unknown))}")
    if objective not in ("reward", "worst_reward", "stability"):
        raise ValueError(f"Unsupported Gym objective: {objective}")


async def evaluate_gym(
    run, policies, *, seeds, features, screening_seeds=(), screening_min_reward=None
):
    """Measure trusted rewards in the sandbox; reuse overlapping cached seeds."""
    seeds, screening_seeds = tuple(seeds), tuple(screening_seeds)
    _check_gym_evaluation(features, seeds, screening_seeds, screening_min_reward)
    results = {}
    if screening_seeds:
        screened = await run.evaluate(policies, seeds=screening_seeds)
        for policy in policies:
            if screened[policy.id] < screening_min_reward:
                results[policy.id] = paper.EvaluationResult(
                    metrics={"reward": screened[policy.id]},
                    feedback=f"Screening reward below {screening_min_reward}",
                    accepted=False,
                )
        policies = [policy for policy in policies if policy.id not in results]
    if policies:
        await run.evaluate(policies, seeds=seeds)
    for policy in policies:
        scores = {seed: score for seed, score in run.scores(policy).items() if seed in seeds}
        mean, std = fmean(scores.values()), pstdev(scores.values())
        descriptors = {"mean_reward": mean, "reward_std": std}
        results[policy.id] = paper.EvaluationResult(
            metrics={"reward": mean, "worst_reward": min(scores.values()), "stability": -std},
            features={name: descriptors[name] for name in features},
            seed_scores=scores,
        )
    return results


def make_environment(name, *, max_steps=None, render_mode=None, shoes_per_episode=24):
    if name in ("Bitcoin", "Blackjack"):
        if render_mode is not None:
            raise ValueError(f"{name} does not support rendering")
        env = (
            BitcoinEnv() if name == "Bitcoin" else BlackjackEnv(shoes_per_episode=shoes_per_episode)
        )
        if max_steps is not None:
            instructions = env.instructions
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
            env.instructions = instructions + f" The episode is truncated after {max_steps} steps."
        return env
    options = {"continuous": True, "enable_wind": True} if name == "LunarLander-v3" else {}
    if max_steps is not None:
        options["max_episode_steps"] = max_steps
    # Keep instructions on a wrapper: Box2D's EzPickle reconstructs the base env.
    env = gym.Wrapper(gym.make(name, render_mode=render_mode, **options))
    env.instructions = (
        f"{name}: {TASKS[name]}\n"
        f"Observation space: {env.observation_space}\nAction space: {env.action_space}\n"
        f"The episode is truncated after {env.spec.max_episode_steps} steps."
    )
    return env


class _ProgressHandler(RichHandler):
    def __init__(self, progress, *, overlap=False):
        super().__init__(
            console=progress.console,
            show_path=False,
            markup=False,
            highlighter=None,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        self.progress = progress
        self.overlap = overlap
        self.generation = progress.add_task("Generating policies", total=0, visible=False)
        self.evaluation = progress.add_task("Evaluating policies", total=0, visible=False)

    def emit(self, record):
        event = getattr(record, "event", None)
        if event == "generation_started":
            if self.overlap:
                total = self.progress.tasks[self.generation].total + record.total
                self.progress.update(self.generation, total=total, visible=True)
            else:
                self.progress.update(self.evaluation, visible=False)
                self.progress.reset(self.generation, total=record.total, visible=True)
        elif event in ("policy_generated", "proposal_discarded"):
            self.progress.advance(self.generation)
        elif event == "evaluation_started":
            if not self.overlap:
                self.progress.update(self.generation, visible=False)
            self.progress.reset(self.evaluation, total=record.total, visible=True)
        elif event in ("policy_evaluated", "evaluation_failed"):
            self.progress.advance(self.evaluation)
        super().emit(record)


def _show_scores(policies, run, console, *, seeds=None):
    table = Table("Policy", "Description", "Score")
    seeds = None if seeds is None else tuple(dict.fromkeys(seeds))
    for policy in policies:
        scores = run.scores(policy)
        values = list(scores.values()) if seeds is None else [scores.get(seed) for seed in seeds]
        score = (
            f"{fmean(values):.1f}"
            if values and all(v is not None for v in values)
            else "unfinished"
        )
        table.add_row(Text(policy.name), Text(policy.description), score)
    console.print(table)


async def run_search(
    generator,
    run,
    *,
    generations,
    batch_size,
    generation_concurrency=4,
    seeds=(0,),
    console=None,
    screening_seeds=(),
    screening_min_reward=None,
    initial_policy=None,
):
    """Display completed policies immediately and keep the same messages in run.log."""
    if isinstance(generator, paper.AlphaEvolve):
        return await run_paper_search(
            generator,
            run,
            generations=generations,
            batch_size=batch_size,
            generation_concurrency=generation_concurrency,
            seeds=seeds,
            console=console,
            screening_seeds=screening_seeds,
            screening_min_reward=screening_min_reward,
            initial_policy=initial_policy,
        )
    seeds = tuple(seeds)
    console = console or Console()
    first_generation = 1
    with run.database() as db:
        if inspect(db.bind).has_table(Generation.__tablename__):
            first_generation = (db.exec(select(func.max(Generation.number))).one() or 0) + 1
    logger = logging.getLogger("rsikit")
    loggers = (logger, logging.getLogger("alphaevolve"))
    old_settings = [(item.level, item.propagate) for item in loggers]
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Generations", total=generations)
        display = _ProgressHandler(progress)
        log = logging.FileHandler(run.path / "run.log", encoding="utf-8")
        log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        for item in loggers:
            item.addHandler(display)
            item.addHandler(log)
            item.setLevel(logging.INFO)
            item.propagate = False
        try:
            logger.info("Run: %s", run.path)
            logger.info("Optimizer: %s", type(generator).__module__)
            for offset in range(generations):
                generation = first_generation + offset
                logger.info("Generation %s/%s", offset + 1, generations)
                history = dict(
                    generation=generation,
                    attempt_start=len(generator.attempts),
                    event_start=len(generator.events),
                    seeds=seeds,
                )
                complete, failures = False, {}
                try:
                    policies = await generator.generate(
                        n=batch_size, concurrency=generation_concurrency
                    )
                    scores = {}
                    while policies:
                        run.save(*history_records(generator, **history, failures=failures))
                        try:
                            scores = await run.evaluate(policies, seeds=seeds)
                            failures = {}
                            break
                        except PolicyError as exc:
                            if not exc.failures:
                                raise
                            failures = exc.failures
                            run.save(*history_records(generator, **history, failures=failures))
                            slots = asyncio.Semaphore(generation_concurrency)

                            async def repair(policy):
                                async with slots:
                                    return await generator.repair(policy, failures[policy.id])

                            failed = {p.id: p for p in policies if p.id in failures}
                            repairs = {
                                id: asyncio.create_task(repair(policy))
                                for id, policy in failed.items()
                            }
                            try:
                                replacements = dict(
                                    zip(repairs, await asyncio.gather(*repairs.values()))
                                )
                            finally:
                                for task in repairs.values():
                                    task.cancel()
                                await asyncio.gather(*repairs.values(), return_exceptions=True)
                            policies = [
                                replacement
                                for policy in policies
                                if (replacement := replacements.get(policy.id, policy)) is not None
                            ]
                    generator.update(
                        scores,
                        seed_scores={
                            policy.id: {
                                seed: score
                                for seed, score in run.scores(policy).items()
                                if seed in seeds
                            }
                            for policy in policies
                        },
                    )
                    complete = True
                finally:
                    run.save(
                        *history_records(generator, **history, complete=complete, failures=failures)
                    )
                _show_scores(policies, run, console)
                if not policies:
                    logger.warning("No surviving policies in this generation; continuing")
                if generator.best is not None:
                    logger.info("Best so far: %s", generator.best.name)
                progress.advance(overall)
        except Exception:
            logger.exception("Run failed; saved results and details are in %s", run.path)
            _show_scores(run.policies(), run, console)
            raise
        finally:
            for item, (level, propagate) in zip(loggers, old_settings):
                item.removeHandler(display)
                item.removeHandler(log)
                item.setLevel(level)
                item.propagate = propagate
            display.close()
            log.close()


async def run_paper_search(
    generator,
    run,
    *,
    generations,
    batch_size,
    generation_concurrency=4,
    seeds=(0,),
    console=None,
    screening_seeds=(),
    screening_min_reward=None,
    initial_policy=None,
):
    """Overlap generation and evaluation; snapshot each evaluated batch.

    Attempts belong to their first-seen snapshot group, not synchronized generations.
    A group's complete flag waits for every assigned attempt to evaluate or discard;
    its island snapshot remains the population at the original batch boundary.
    """
    seeds, screening_seeds = tuple(seeds), tuple(screening_seeds)
    _check_gym_evaluation(
        generator.config.features,
        seeds,
        screening_seeds,
        screening_min_reward,
        generator.config.objective,
    )
    console = console or Console()
    logger = logging.getLogger("rsikit")
    loggers = (logger, logging.getLogger("alphaevolve"))
    old_settings = [(item.level, item.propagate) for item in loggers]
    generation = 1
    with run.database() as db:
        if inspect(db.bind).has_table(Generation.__tablename__):
            generation = (db.exec(select(func.max(Generation.number))).one() or 0) + 1
    next_attempt = len(generator.attempts)
    event_start = len(generator.events)
    unresolved = {}
    failures = {}
    failed_groups = set()
    failed_versions = {}

    async def evaluate(policies):
        nonlocal failures
        try:
            return await evaluate_gym(
                run,
                policies,
                seeds=seeds,
                features=generator.config.features,
                screening_seeds=screening_seeds,
                screening_min_reward=screening_min_reward,
            )
        except PolicyError as exc:
            failures = exc.failures
            raise

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Evaluated policies", total=generations * batch_size)
        display = _ProgressHandler(progress)
        log = logging.FileHandler(run.path / "run.log", encoding="utf-8")
        log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        for item in loggers:
            item.addHandler(display)
            item.addHandler(log)
            item.setLevel(logging.INFO)
            item.propagate = False

        def persist(event, policies):
            nonlocal next_attempt, event_start, generation
            for record in generator.attempts[next_attempt:]:
                unresolved[record["id"]] = (generation, record)
            next_attempt = len(generator.attempts)
            batches = {generation: []}
            for number, record in unresolved.values():
                batches.setdefault(number, []).append(record)
                policy = record.get("policy")
                version = (record["id"], record.get("revision", 0))
                if event == "evaluation_failed" and policy is not None and policy.id in failures:
                    failed_versions[version] = failures[policy.id]
                if record["status"] in ("cancelled", "rejected", "error"):
                    failed_groups.add(number)
            for number, attempts in batches.items():
                complete = (
                    (bool(attempts) or event == "evaluated")
                    and number not in failed_groups
                    and all(row["status"] in ("evaluated", "discarded") for row in attempts)
                )
                rows = history_records(
                    generator,
                    generation=number,
                    attempt_start=0,
                    attempts=attempts,
                    event_start=event_start,
                    seeds=seeds,
                    complete=complete,
                    failures={
                        row["policy"].id: failed_versions[version]
                        for row in attempts
                        if (version := (row["id"], row.get("revision", 0))) in failed_versions
                    },
                )
                if number != generation:
                    # Finish the older group without replacing its population snapshot.
                    rows.pop()
                    if complete:
                        with run.database() as db:
                            snapshot = db.get(Generation, number)
                        snapshot.complete = True
                        rows.append(snapshot)
                run.save(*rows)
            unresolved_keys = [
                key
                for key, (_, row) in unresolved.items()
                if row["status"] not in ("generating", "generated", "repaired", "evaluating")
            ]
            for key in unresolved_keys:
                del unresolved[key]
            if event == "evaluated":
                _show_scores(policies, run, console, seeds=seeds)
                progress.advance(overall, len(policies))
                if generator.best is not None:
                    logger.info("Best so far: %s", generator.best.name)
                generation += 1
                event_start = len(generator.events)

        try:
            logger.info("Run: %s", run.path)
            logger.info("Optimizer: %s", type(generator).__module__)
            if initial_policy is not None:
                source = initial_policy.read_text(encoding="utf-8")
                check_program(source)
                policy = _policy_class(initial_policy.stem, source, "User-supplied initial policy")
                result = (await evaluate([policy]))[policy.id]
                if not result.accepted:
                    raise ValueError("Initial policy failed screening; it was not registered")
                generator.register_initial(policy, result)
                _show_scores([policy], run, console, seeds=seeds)
            await paper.search(
                generator,
                evaluate,
                proposals=generations * batch_size,
                generation_concurrency=generation_concurrency,
                evaluation_batch_size=batch_size,
                on_event=persist,
            )
        except Exception:
            logger.exception("Run failed; saved results and details are in %s", run.path)
            raise
        finally:
            try:
                if unresolved or next_attempt < len(generator.attempts):
                    persist("finished", [])
                generator.checkpoint()
            finally:
                for item, (level, propagate) in zip(loggers, old_settings):
                    item.removeHandler(display)
                    item.removeHandler(log)
                    item.setLevel(level)
                    item.propagate = propagate
                display.close()
                log.close()


def _resume_settings(parser, args):
    """Restore resolved settings, allowing only budget and worker overrides."""
    if args.output or args.initial_policy or args.search_config:
        parser.error(
            "--resume cannot be combined with --output, --initial-policy or --search-config"
        )
    try:
        saved = json.loads((args.resume / "experiment.json").read_text(encoding="utf-8"))
        if saved["variant"] != "paper":
            raise ValueError("Only paper runs have resumable optimizer checkpoints")
        population = args.resume / "population.sqlite"
        if not population.is_file():
            raise ValueError("Run has no population checkpoint (population.sqlite)")
        with closing(sqlite3.connect(population.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute("SELECT value FROM state WHERE key = 'optimizer'").fetchone()
            state = json.loads(row[0]) if row is not None else None
            if (
                not isinstance(state, dict)
                or not {"task", "context", "completed", "rng"} <= state.keys()
            ):
                raise ValueError("Population checkpoint has no saved optimizer state")
        locked = (
            "variant",
            "search_seed",
            "env",
            "model",
            "ensemble",
            "max_steps",
            "seeds",
            "screening_seeds",
            "screening_min_reward",
        )
        defaults = {
            name: saved[name]
            for name in (
                *locked,
                "generations",
                "batch_size",
                "generation_concurrency",
                "concurrency",
            )
        }
        defaults.update(mode=saved["config"]["mode"], max_repairs=saved["config"]["max_repairs"])
        parser.set_defaults(**defaults)
        args = parser.parse_args()
        for name in (*locked, "mode", "max_repairs"):
            if getattr(args, name) != defaults[name]:
                raise ValueError(f"--{name.replace('_', '-')} cannot change when resuming a run")
    except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as exc:
        parser.error(f"Cannot resume run: {exc}")
    return args, saved


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("paper", "original", "improved"), default="paper")
    parser.add_argument("--search-seed", type=int, default=0, help="Optimizer sampling seed")
    parser.add_argument("--env", choices=TASKS, default="CartPole-v1")
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument(
        "--generations",
        type=int,
        default=25,
        help="Proposal batches to attempt; additional batches when resuming (default: 25)",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-repairs", type=int, help="Override Config.max_repairs (default: 2)")
    parser.add_argument("--mode", choices=("diff", "rewrite"))
    parser.add_argument("--search-config", type=Path, help="JSON Config fields; CLI overrides win")
    parser.add_argument(
        "--ensemble", nargs=2, action="append", default=[], metavar=("MODEL", "WEIGHT")
    )
    parser.add_argument(
        "--initial-policy", type=Path, help="Paper variant: evaluate and seed a policy"
    )
    parser.add_argument("--screening-seeds", type=int, nargs="+")
    parser.add_argument("--screening-min-reward", type=float)
    parser.add_argument(
        "--generation-concurrency",
        type=int,
        default=4,
        help="Concurrent model proposals and runtime repairs (default: 4)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Concurrent sandbox evaluations (default: 4)"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Override the environment's native time limit"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Continue a paper run using its saved settings")
    args = parser.parse_args()
    saved = None
    if args.resume is not None:
        args, saved = _resume_settings(parser, args)
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    if (
        args.generations < 0
        or min(args.batch_size, args.generation_concurrency, args.concurrency) < 1
    ):
        parser.error(
            "Generations must be nonnegative; batch size and worker counts must be positive"
        )

    if bool(args.screening_seeds) != (args.screening_min_reward is not None):
        parser.error("--screening-seeds and --screening-min-reward must be supplied together")
    if args.screening_min_reward is not None and not math.isfinite(args.screening_min_reward):
        parser.error("--screening-min-reward must be finite")
    if args.variant != "paper" and (args.initial_policy or args.screening_seeds):
        parser.error("Initial policies and screening require --variant paper")
    variant = {"paper": paper, "original": original, "improved": improved}[args.variant]
    try:
        options = (
            dict(saved["config"])
            if saved is not None
            else {}
            if args.search_config is None
            else json.loads(args.search_config.read_text())
        )
        if not isinstance(options, dict):
            raise ValueError("Search config must be a JSON object")
        options["mode"] = args.mode or options.get(
            "mode", "rewrite" if args.env == "BipedalWalker-v3" else "diff"
        )
        if args.max_repairs is not None:
            options["max_repairs"] = args.max_repairs
        if args.variant == "paper":
            options.setdefault("features", FEATURE_BOUNDS[args.env])
        config = variant.Config(**options)
        if args.variant == "paper":
            _check_gym_evaluation(
                config.features,
                args.seeds,
                args.screening_seeds,
                args.screening_min_reward,
                config.objective,
            )
        models = [(args.model, 1.0)] + [(name, float(weight)) for name, weight in args.ensemble]
        if any(not math.isfinite(weight) or weight <= 0 for _, weight in models):
            raise ValueError("Ensemble weights must be positive and finite")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.mode = config.mode
    prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent
    executor = Executor(concurrency=args.concurrency)
    async with AsyncExitStack() as stack:
        environment = stack.enter_context(make_environment(args.env, max_steps=args.max_steps))
        run = await stack.enter_async_context(
            Run.open(args.resume, environment=environment, executor=executor)
            if args.resume
            else Run.create(
                name=f"{args.env.lower()}-{args.variant}-evolution",
                environment=environment,
                executor=executor,
                path=args.output,
            )
        )
        providers = [
            (OpenRouterAPI(model=name, max_output_tokens=8192, timeout=120), weight)
            for name, weight in models
        ]
        generator = variant.AlphaEvolve(
            task="Maximize cumulative episode reward in the described environment.",
            context=environment.instructions,
            provider=providers[0][0],
            ensemble=providers,
            config=config,
            seed=args.search_seed,
            **(
                {"database_path": run.path / "population.sqlite"} if args.variant == "paper" else {}
            ),
        )
        if saved is None:
            (run.path / "experiment.json").write_text(
                json.dumps({**vars(args), "config": asdict(config)}, default=str, indent=2) + "\n",
                encoding="utf-8",
            )
        try:
            await run_search(
                generator,
                run,
                generations=args.generations,
                batch_size=args.batch_size,
                generation_concurrency=args.generation_concurrency,
                seeds=args.seeds,
                screening_seeds=args.screening_seeds or (),
                screening_min_reward=args.screening_min_reward,
                initial_policy=args.initial_policy,
            )
        except Exception:
            # run_search has already displayed and saved the traceback.
            raise SystemExit(1) from None
        finally:
            if args.variant == "paper":
                generator.close()


if __name__ == "__main__":
    asyncio.run(main())
