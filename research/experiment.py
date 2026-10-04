"""Shared configuration and lifecycle for environment-owned evaluations."""

import asyncio
import contextvars
import hashlib
import inspect
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Callable
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field

from research.rewards import episode_error, episode_scores
from rsikit import Policy, Run


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class EvaluationConfig(Options):
    seeds: list[int] = Field(default_factory=lambda: list(range(10)))
    workers: int = Field(default=4, ge=1)
    timeout_per_seed: float = Field(default=60, gt=0)
    batch_size: int | None = Field(default=None, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    score_key: str = "return"


@dataclass(frozen=True)
class EnvironmentDefinition:
    Options: type[BaseModel]
    add_arguments: Callable
    describe: Callable
    open_evaluator: Callable
    evaluation_defaults: dict = field(default_factory=dict)
    supported_evaluation_fields: frozenset = frozenset({"max_steps", "score_key"})
    score_keys: tuple[str, ...] = ("return",)
    protocol: str = "gymnasium-episode-v1"
    provenance: Callable = dict


EXECUTION = contextvars.ContextVar("experiment_execution", default=None)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record_execution(policy_id, *, seed_runs, steps=None):
    """Backend evidence, separate from callback requests and cached measurements."""
    scope = EXECUTION.get()
    if scope is None:
        return
    counts, request = scope
    counts["_executed"].add((request, policy_id))
    counts["evaluation_submissions"] = len(counts["_executed"])
    for key, value in (("seed_runs", seed_runs), ("transitions", steps)):
        if value is None:
            counts[key] = None
        elif counts[key] is not None:
            counts[key] += value


class _EpisodeEvidence(logging.Handler):
    def emit(self, record):
        if getattr(record, "event", None) == "episode_timing":
            record_execution(record.policy_id, seed_runs=1)


def _new_counts():
    return dict(
        requested_candidates=0,
        requested_seed_runs=0,
        evaluation_submissions=0,
        seed_runs=0,
        transitions=0,
        failures=0,
        _executed=set(),
    )


def _evidence(episodes):
    """Compact reporting only; optimizers receive the original episodes."""
    return dict(
        scores=episode_scores(episodes),
        failure=episode_error(episodes),
        accepted=bool(episodes) and episode_error(episodes) is None,
    )


def _complete(measurement, seeds):
    return (
        bool(measurement) and episode_error(measurement) is None and set(measurement) == set(seeds)
    )


def _generation_summary(provider):
    events = provider.events
    usage = [e.get("usage") for e in events]

    def total(key):
        values = [u.get(key) if isinstance(u, dict) else None for u in usage]
        return sum(values) if all(v is not None for v in values) else None

    costs = [e.get("actual_cost") for e in events]
    return dict(
        calls=provider.calls,
        input_tokens=total("prompt_tokens"),
        output_tokens=total("completion_tokens"),
        actual_cost=sum(costs) if all(c is not None for c in costs) else None,
        reserved_tokens=provider.reserved_tokens,
        reserved_cost=provider.reserved_cost,
        stop=provider.stopped,
        proposals=len({e["attempt"] for e in events if "attempt" in e}) or None,
        repairs=sum(bool(e.get("repair")) for e in events),
    )


async def select_winner(candidates, evaluate, run, selection):
    candidates = list(dict.fromkeys(candidates))[: selection["finalists"]]
    seeds = selection["validation_seeds"]
    validation = await evaluate(candidates, seeds, "validation") if seeds and candidates else {}
    valid = [p for p in candidates if _complete(validation[p.id], seeds)] if seeds else candidates
    if not valid:
        save_json(run.path / "selection.json", dict(winner=None, status="no_valid_candidate"))
        return dict(
            status="no_valid_candidate",
            winner=None,
            validation={k: _evidence(v) for k, v in validation.items()},
            test=None,
        )
    winner = (
        max(valid, key=lambda p: fmean(episode_scores(validation[p.id]).values()))
        if seeds
        else valid[0]
    )
    winner.to_file(run.path / "winner.py")
    result = dict(
        winner=winner.id,
        validation={k: _evidence(v) for k, v in validation.items()} if seeds else "not evaluated",
        test="not evaluated",
    )
    save_json(run.path / "selection.json", result)
    seeds = selection["test_seeds"]
    if seeds:
        measured = (await evaluate([winner], seeds, "test"))[winner.id]
        result["test"] = _evidence(measured)
        result["test_mean"] = (
            fmean(episode_scores(measured).values()) if _complete(measured, seeds) else None
        )
        if not _complete(measured, seeds):
            result["status"] = "evaluation_failed"
    return result


async def _experiment(config, *, resume=False):
    from research.cli import load_component, parse_config

    # The Python entry point receives the same validation as the CLI.
    config = parse_config([config["command"]], config=config)
    definition = load_component(config["env"], kind="environment", base_dir=Path.cwd())
    env_options = definition.Options.model_validate_json(json.dumps(config["environment"]))
    evaluation = EvaluationConfig(**config["evaluation"])
    task = definition.describe(env_options, evaluation)
    provenance = definition.provenance()  # Includes Ocean's pin/binary check before model calls.
    search = config["command"] == "run"
    optimizer = (
        load_component(config["optimizer"], kind="optimizer", base_dir=Path.cwd())
        if search
        else None
    )
    policies = [Policy.from_file(p) for p in config.get("policies", [])]
    if len({p.id for p in policies}) != len(policies):
        raise ValueError("Duplicate policy files/IDs")
    if search and not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("Set OPENROUTER_API_KEY before running paid generation")
    manifest = dict(
        protocol=definition.protocol,
        environment=config["env"],
        python=platform.python_version(),
        platform=platform.platform(),
        provenance=provenance,
        image=os.environ.get("RSIKIT_IMAGE_ID"),
        revision=os.environ.get("RSIKIT_GIT_REVISION"),
        seed_semantics="one episode per seed",
        max_transitions_per_candidate=len(evaluation.seeds) * evaluation.max_steps
        if evaluation.max_steps
        else None,
        source_hashes={},
    )
    if search:
        manifest["optimization_schedule"] = "round-v1"
    manifest["versions"] = {}
    for package in ("rsikit", "gymnasium", "numpy", "slick-ai"):
        try:
            manifest["versions"][package] = version(package)
        except PackageNotFoundError:
            manifest["versions"][package] = None
    evaluator_factory = getattr(definition.open_evaluator, "func", definition.open_evaluator)
    for name, function in (
        ("environment_integration", evaluator_factory),
        ("optimizer_integration", optimizer.optimize if optimizer else None),
    ):
        source = inspect.getsourcefile(function) if function else None
        if source and Path(source).is_file():
            manifest["source_hashes"][name] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    for name, value in [
        ("env", config["env"]),
        ("optimizer", config.get("optimizer", "")),
        *[(f"policy:{i}", value) for i, value in enumerate(config.get("policies", []))],
    ]:
        if value.endswith(".py"):
            manifest["source_hashes"][name] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    counts = {}
    summary = dict(status="running", phases={})
    provider = None
    with (
        Run.open(config["output"])
        if resume
        else Run.create(
            name=f"{config['env']}-{config.get('optimizer', 'evaluate')}", path=config["output"]
        )
    ) as run:
        previous = {}
        previous_calls = []
        if resume:
            if (run.path / "summary.json").exists():
                previous = json.loads((run.path / "summary.json").read_text())
            if (run.path / "model_calls.jsonl").exists():
                with (run.path / "model_calls.jsonl").open() as stream:
                    previous_calls = [json.loads(line) for line in stream]
            if max((e["call"] for e in previous_calls), default=0) < previous.get(
                "generation", {}
            ).get("calls", 0):
                raise ValueError("Model call ledger is incomplete; cannot restore budget")
        else:
            (run.path / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
            )
            save_json(run.path / "manifest.json", manifest)
        # Retain the metadata expected by the existing generation-video exporter.
        if search and config["videos"]["top"]:
            save_json(
                run.path / "experiment.json",
                dict(
                    env="Blackjack",
                    seeds=evaluation.seeds,
                    heldout_seeds=config["selection"]["test_seeds"],
                    shoes_per_seed=config["environment"]["shoes_per_episode"],
                    max_steps=evaluation.max_steps,
                    episode_timeout=evaluation.timeout_per_seed,
                ),
            )
        handler = _EpisodeEvidence()
        logger = logging.getLogger("rsikit.execution")
        previous_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            async with definition.open_evaluator(
                options=env_options, evaluation=evaluation, run=run
            ) as evaluate:

                async def measured(policies, seeds, phase):
                    policies, seeds = list(policies), list(seeds)
                    if not policies:
                        return {}
                    if len({p.id for p in policies}) != len(policies):
                        raise ValueError("Duplicate policies in evaluation request")
                    tally = counts.setdefault(phase, _new_counts())
                    tally["requested_candidates"] += len(policies)
                    tally["requested_seed_runs"] += len(policies) * len(seeds)
                    for p in policies:
                        run.save_policy(p)
                    token = EXECUTION.set((tally, uuid4().hex))
                    try:
                        results = await evaluate(policies, seeds)
                        if set(results) != {p.id for p in policies}:
                            raise ValueError(
                                "Evaluator must return exactly the requested policy IDs"
                            )
                        for p in policies:
                            result = results[p.id]
                            if (bool(result) and episode_error(result) is None) and not _complete(
                                result, seeds
                            ):
                                raise ValueError(
                                    "Evaluator returned an incomplete accepted seed panel"
                                )
                            if bool(result) and episode_error(result) is None:
                                run.save_policy(p, scores=episode_scores(result))
                            else:
                                tally["failures"] += 1
                        with (run.path / "measurements.jsonl").open("a") as stream:
                            stream.write(
                                json.dumps(
                                    dict(
                                        phase=phase,
                                        seeds=seeds,
                                        results={k: _evidence(v) for k, v in results.items()},
                                    ),
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                        return results
                    finally:
                        EXECUTION.reset(token)

                if search:
                    from research.providers import BudgetProvider, UsageOpenRouter

                    generation = config["generation"]
                    raw = UsageOpenRouter(
                        model=config["model"],
                        max_output_tokens=generation["max_output_tokens"],
                        timeout=generation["timeout"],
                        max_retries=0,
                    )

                    def log(event):
                        with (run.path / "model_calls.jsonl").open("a") as stream:
                            stream.write(json.dumps(event, allow_nan=False) + "\n")

                    provider = BudgetProvider(
                        raw,
                        **config["budget"],
                        max_input_tokens=generation["max_input_tokens"],
                        max_output_tokens=generation["max_output_tokens"],
                        log=log,
                    )
                    provider.restore(previous_calls)
                    # Shared runtime settings are available to all optimizer entry points.
                    options = {
                        **config["optimizer_options"],
                        "generation": generation,
                        "videos": config["videos"],
                    }
                    candidates = await optimizer.optimize(
                        task=task,
                        provider=provider,
                        evaluate=lambda ps: measured(ps, evaluation.seeds, "search"),
                        run=run,
                        options=options,
                        seed=config["search_seed"],
                    )
                    summary["status"] = "budget_exhausted" if provider.stopped else "completed"
                    # Only candidates with successful search evidence may become finalists.
                    candidates = [
                        p
                        for p in candidates
                        if all(run.scores(p).get(s) is not None for s in evaluation.seeds)
                    ]
                    summary.update(
                        await select_winner(candidates, measured, run, config["selection"])
                    )
                else:
                    results = await measured(policies, evaluation.seeds, "evaluation")
                    summary.update(
                        status="completed"
                        if all((bool(r) and episode_error(r) is None) for r in results.values())
                        else "evaluation_failed",
                        measurements={k: _evidence(v) for k, v in results.items()},
                    )
        except asyncio.CancelledError:
            summary["status"] = "cancelled"
            raise
        except BaseException as exc:
            summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
            summary["phases"] = {
                phase: {k: v for k, v in tally.items() if not k.startswith("_")}
                for phase, tally in counts.items()
            }
            # Custom evaluators can report actual work through record_execution; absent evidence is unknown.
            if config["env"].endswith(".py"):
                for phase in summary["phases"].values():
                    if not phase["evaluation_submissions"]:
                        for key in ("evaluation_submissions", "seed_runs", "transitions"):
                            phase[key] = None
            for phase, prior in previous.get("phases", {}).items():
                if phase == "rendering":
                    continue  # Rendering is recounted from its full append-only log below.
                current = summary["phases"].setdefault(phase, {})
                for key, value in prior.items():
                    added = current.get(key, 0)
                    current[key] = None if value is None or added is None else value + added
            replay_log = run.path / "videos/executions.jsonl"
            if replay_log.exists():
                replay = [json.loads(line) for line in replay_log.read_text().splitlines()]
                summary["phases"]["rendering"] = dict(
                    evaluation_submissions=len({e["policy_id"] for e in replay}),
                    seed_runs=len(replay),
                    transitions=None,
                )
            if provider is not None:
                summary["generation"] = _generation_summary(provider)
            save_json(run.path / "summary.json", summary)
    return summary


async def run_experiment(config: dict) -> dict:
    if config.get("command") != "run":
        raise ValueError("run_experiment requires a run configuration")
    return await _experiment(config)


async def resume_experiment(path: str | Path) -> dict:
    """Continue an Elite search in place using its saved configuration and budget."""
    from research.cli import read_config

    path = Path(path).resolve()
    config = read_config(path / "config.yaml")
    if config.get("command") != "run" or config.get("optimizer") != "elite":
        raise ValueError("Resume currently supports unified CLI Elite searches only")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("optimization_schedule") != "round-v1":
        from sqlalchemy import inspect as inspect_database
        from sqlmodel import select

        from research.elitesearch.records import Generation

        with Run.open(path) as run, run.database() as db:
            if inspect_database(db.get_bind()).has_table(Generation.__tablename__):
                if any(row.status != "completed" for row in db.exec(select(Generation))):
                    raise ValueError(
                        "Cannot resume an incomplete legacy streaming checkpoint; start a new run from an exported policy"
                    )
        changes = path / "schedule_changes.jsonl"
        if not changes.exists():
            with changes.open("a") as stream:
                stream.write(
                    json.dumps({"from": manifest.get("optimization_schedule"), "to": "round-v1"})
                    + "\n"
                )
    config["output"] = str(path)
    return await _experiment(config, resume=True)


async def evaluate_policies(config: dict) -> dict:
    if config.get("command") != "evaluate":
        raise ValueError("evaluate_policies requires an evaluate configuration")
    return await _experiment(config)
