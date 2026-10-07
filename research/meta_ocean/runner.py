"""Trusted GEPA campaign coordinator. Run locally with access to the Docker daemon."""

import argparse
import asyncio
import os
import subprocess
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from time import monotonic

import yaml

from research.cli import read_config
from research.experiment import EvaluationConfig
from research.ocean.baselines import policies
from research.ocean.environment import definition, describe, metadata
from research.ocean.evaluator import rollout
from rsikit.evaluation import PolicyError
from rsikit.progress import bind_run

from .checkpoints import archive, read_json, run_lock, save_json, upgrade_candidates
from .experiment import (
    Config,
    Trial,
    append_json,
    digest,
    progress,
    provider,
    scorecard,
    suite_scorecard,
    usage,
)
from .gepa import optimize
from .sandbox import RemotePolicy, run_controller


async def evaluate_policy(
    config, source, seeds, path, *, environment="g2048", show_progress=True, label_prefix=""
):
    def event(message, **payload):
        progress(message, **payload if show_progress else {})

    task = definition(environment)
    label = f"{environment}/{path.parent.name}/{path.stem}"
    if path.parent.name == environment:
        label = f"calibration/{environment}/{path.stem}"
    if label_prefix:
        label = f"{label_prefix}/{label}"
    event(
        f"{label}: starting {len(seeds)} games, at most {config.max_steps} steps each",
        kind="batch_started",
        batch_id=str(path),
        label=f"Batch {label}",
        total_candidates=1,
    )
    candidate = dict(
        kind="candidate",
        batch_id=str(path),
        attempt_id="0",
        revision=0,
        proposal_done=True,
        name=label,
        policy_id=digest(source)[:12],
    )
    event(f"{label}: evaluating", **candidate, status="evaluating")

    def completed(rows):
        event(
            f"{label}: {len(rows)}/{len(seeds)} games; "
            f"{sum(row['steps'] for row in rows)} transitions"
        )

    try:
        result = await asyncio.wait_for(
            rollout(
                source,
                seeds,
                config.batch_size,
                config.max_steps,
                env_name=environment,
                score_key=task.evaluation_defaults["score_key"],
                env_kwargs=task.Options().model_dump(),
                _policy_factory=partial(RemotePolicy, image=config.image),
                _on_batch=completed,
            ),
            config.panel_timeout,
        )
        save_json(path, result)
        scores = [row["score"] for row in result["results"]]
        if len(scores) != len(seeds):
            raise RuntimeError("Incomplete episode panel")
    except BaseException as exc:
        event(
            f"{label}: {type(exc).__name__}: {exc}",
            **candidate,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    event(
        f"{label}: mean score {fmean(scores):.2f}",
        **candidate,
        status="evaluated",
        score=fmean(scores),
    )
    return dict(score=fmean(scores), scores=scores, steps=result["steps"])


async def run_trial(config, source, path, replicate, *, environment="g2048", model=None):
    started = monotonic()
    path.mkdir(parents=True, exist_ok=False)
    (path / "controller.py").write_text(source)
    model = model if model is not None else provider(config, path / "model_calls.jsonl")
    starter = policies(environment)[0].source
    count = 0

    async def evaluate(candidate, seeds):
        nonlocal count
        count += 1
        return await evaluate_policy(
            config, candidate, seeds, path / f"panel-{count}.json", environment=environment
        )

    trial = Trial(config, path, starter, evaluate, model, replicate)
    definition_ = definition(environment)
    task = describe(
        environment,
        definition_.Options(),
        EvaluationConfig(
            max_steps=config.max_steps,
            batch_size=config.batch_size,
            score_key=definition_.evaluation_defaults["score_key"],
        ),
    )
    context = dict(
        task=task,
        starter=starter,
        seed=replicate,
        budget=dict(evaluations=config.evaluations, output_tokens=config.output_tokens),
    )
    error = None
    try:
        await run_controller(
            source, context, trial.handle, image=config.image, timeout=config.trial_timeout
        )
    except (PolicyError, asyncio.TimeoutError, BrokenPipeError, ConnectionResetError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        row = trial.finish(error)
        row["search_seconds"] = monotonic() - started
    # The controller is gone before private seeds or scores are accessed.
    row.update(environment=environment, replicate=replicate, policy_hash=digest(trial.incumbent))
    save_json(path / "commit.json", row)
    try:
        audit = await evaluate_policy(
            config,
            trial.incumbent,
            list(range(replicate * 10000 + 1000, replicate * 10000 + 1000 + config.audit_cases)),
            path / "audit.json",
            environment=environment,
        )
        row.update(score=audit["score"], audit_transitions=audit["steps"], audit_error=None)
    except (PolicyError, asyncio.TimeoutError, ValueError) as exc:
        row.update(score=0, audit_transitions=None, audit_error=f"{type(exc).__name__}: {exc}")
    row["total_seconds"] = monotonic() - started
    save_json(path / "summary.json", row)
    return row


async def benchmark(config, source, path, replicates, anchors):
    path.mkdir(parents=True, exist_ok=False)
    rows = []
    for environment in config.environments:
        for replicate in replicates:
            progress(f"{path.parent.name}/{path.name}: {environment} replicate {replicate}")
            rows.append(
                await run_trial(
                    config,
                    source,
                    path / environment / str(replicate),
                    replicate,
                    environment=environment,
                )
            )
    result = suite_scorecard(rows, anchors, config.objective)
    result["trials"] = rows
    result["controller_hash"] = digest(source)
    save_json(path / "scorecard.json", result)
    progress(f"{path.name}: S={result['S']:.2f}, T={result['T']}, E={result['E']}")
    return result


async def calibrate(config, path):
    path.mkdir()
    seeds = list(range(2_000_000_000, 2_000_000_000 + config.calibration_cases))
    anchors = {}
    for environment in config.environments:
        directory = path / environment
        directory.mkdir()
        anchors[environment] = anchor = {}
        candidates = policies(environment)
        for name, policy in (("baseline", candidates[0]), ("reference", candidates[-1])):
            policy.to_file(directory / f"{name}.py")
            result = await evaluate_policy(
                config,
                policy.source,
                seeds,
                directory / f"{name}.json",
                environment=environment,
            )
            anchor[name] = result["score"]
        # All anchors freeze before paid generation; reject degenerate scales.
        scorecard(
            [dict(score=anchor["baseline"], output_tokens=0, evaluations=0)],
            **anchor,
            objective="performance",
        )
    save_json(path / "anchors.json", anchors)
    return anchors


def campaign(config, path, anchors, loop):
    path.mkdir()
    starter = Path(__file__).with_name("controller.py").read_text()
    editor = provider(config, path / "editor_calls.jsonl", editor=True)
    calls = 0
    summary = dict(status="running", objective=config.objective)

    def measure(source):
        nonlocal calls
        calls += 1
        result = loop.run_until_complete(
            benchmark(config, source, path / f"development-{calls}", config.development, anchors)
        )
        append_json(path / "benchmarks.jsonl", dict(number=calls, **result))
        return result

    def edit(prompt):
        return loop.run_until_complete(editor.acall(prompt))[0]

    try:
        best = optimize(starter, measure, edit, config, path)
        winner = best.pop("controller")
        (path / "winner.py").write_text(winner)
        # Freeze the winner before either held-out phase. Neither can trigger reselection.
        save_json(path / "selection.json", dict(controller_hash=digest(winner), **best))
        summary.update(best, controller_hash=digest(winner), heldout={})
        for phase in ("validation", "test"):
            phase_config = config.model_copy(
                update={
                    "audit_cases": config.final_audit_cases
                    if phase == "test"
                    else config.audit_cases
                }
            )
            summary["heldout"][phase] = {}
            for name, source in (("baseline", starter), ("winner", winner)):
                summary["heldout"][phase][name] = loop.run_until_complete(
                    benchmark(
                        phase_config,
                        source,
                        path / f"{phase}-{name}",
                        getattr(config, phase),
                        anchors,
                    )
                )
        summary["status"] = "completed"
        return summary
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["editor"] = usage(editor)
        save_json(path / "summary.json", summary)


def workload(config):
    if getattr(config, "optimizer", "gepa") == "elitetable":
        from .elitetable import workload as elite_workload

        return elite_workload(config)
    tracks = 3 if config.objective == "all" else 1
    tasks = len(config.environments)
    trials = (
        tracks
        * tasks
        * (
            config.benchmark_limit * len(config.development)
            + 2 * (len(config.validation) + len(config.test))
        )
    )
    return dict(
        search_games_per_evaluation=10,
        max_transitions_per_evaluation=10 * config.max_steps,
        max_controller_trials=trials,
        max_evaluated_proposals=trials * config.evaluations,
        max_search_games=trials * config.evaluations * 10,
        private_audit_games=tracks
        * tasks
        * (
            (config.benchmark_limit * len(config.development) + 2 * len(config.validation))
            * config.audit_cases
            + 2 * len(config.test) * config.final_audit_cases
        ),
        calibration_games=tasks * 2 * config.calibration_cases,
        max_model_reservation_usd=trials * config.trial_spend_cap
        + tracks * config.editor_spend_cap,
    )


def run(config, output, *, resume=False):
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("Set OPENROUTER_API_KEY for the metered generation gateway")
    if resume and getattr(config, "optimizer", "gepa") != "elitetable":
        raise ValueError("--resume currently supports the EliteTable meta experiment only")
    if resume and not (output / "manifest.json").exists():
        raise ValueError("--resume requires an existing run with manifest.json")
    output.mkdir(parents=True, exist_ok=resume)
    optimizer = "EliteTable" if getattr(config, "optimizer", "gepa") == "elitetable" else "GEPA"
    with run_lock(output), bind_run(output):
        progress(
            "Ocean mixed-task meta experiment",
            kind="environment",
            name="Ocean " + ", ".join(config.environments),
        )
        progress(
            f"Starting {optimizer} meta experiment; checking Ocean and Docker",
            kind="search_started",
            optimizer=f"{optimizer} meta",
            total_candidates=None,
        )
        result = _run(config, output, resume=resume)
        progress(
            f"{optimizer} meta experiment completed",
            kind="search_finished",
            status="completed",
            reason="all campaigns finished",
        )
        return result


def _run(config, output, *, resume=False):
    elite = getattr(config, "optimizer", "gepa") == "elitetable"
    run_campaign = campaign
    if elite:
        from .elitetable import campaign as run_campaign
    previous = read_json(output / "manifest.json") if resume else None
    upgrading_repairs = resume and previous["protocol"] in (
        "ocean-elitetable-meta-v2",
        "ocean-elitetable-meta-v3",
    )
    if resume:
        saved_config = yaml.safe_load((output / "config.yaml").read_text())
        if upgrading_repairs:
            # Explicit protocol migration adds the repair allowance to the evaluation
            # ceiling. It does not silently increase token or spend caps.
            saved_config = type(config).model_validate(saved_config).model_dump()
        supplied = config.model_dump()
        # Tags can move; a resumed run always uses the original immutable image.
        supplied["image"] = saved_config["image"]
        worker_settings = {"trial_workers", "evaluation_workers", "model_workers"}
        if {k: v for k, v in supplied.items() if k not in worker_settings} != {
            k: v for k, v in saved_config.items() if k not in worker_settings
        }:
            raise ValueError(
                "Resume config differs from the saved run; keep all experiment settings fixed"
            )
        config = config.model_copy(update={"image": saved_config["image"]})
        if upgrading_repairs:
            config = config.model_copy(update={"max_repairs": max(5, config.max_repairs)})
    native = {name: metadata(name) for name in config.environments}
    if "maze" in config.environments:
        from research.ocean.maze import verify_splits

        progress("Checking Maze map identities across calibration, search and audit splits")
        splits = verify_splits(config)
        if resume and read_json(output / "maze_splits.json") != splits:
            raise ValueError("Maze splits changed since the original run")
        if not resume:
            save_json(output / "maze_splits.json", splits)
    image = subprocess.run(
        ["docker", "image", "inspect", config.image, "--format", "{{.Id}}"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    config = config.model_copy(update={"image": image})  # Pin the inspected image for all workers.
    manifest = dict(
        protocol="ocean-elitetable-meta-v7" if elite else "ocean-gepa-meta-suite-v2",
        optimizer="elitetable" if elite else "gepa",
        gepa=None if elite else version("gepa"),
        native=native,
        tasks={
            name: dict(
                options=definition(name).Options().model_dump(),
                score_key=definition(name).evaluation_defaults["score_key"],
            )
            for name in config.environments
        },
        worker_image=image,
        workload=workload(config),
        source_hashes={p.name: digest(p.read_text()) for p in Path(__file__).parent.glob("*.py")},
        scope="Equal-weight improvement over fixed EliteTable"
        if elite
        else "Equal-weight environment suite; frozen per-task anchors",
        baseline_sources={
            str(p.relative_to(Path(__file__).parent.parent)): digest(p.read_text())
            for p in (Path(__file__).parent.parent / "elitesearch").rglob("*")
            if p.suffix in (".py", ".j2")
        }
        if elite
        else {},
    )
    if resume:
        for key in ("optimizer", "native", "tasks", "worker_image", "baseline_sources"):
            saved, current = previous[key], manifest[key]
            if key == "baseline_sources":
                # Standalone replay tooling is never imported by the meta baseline.
                saved = {k: v for k, v in saved.items() if k != "elitesearch/videos.py"}
                current = {k: v for k, v in current.items() if k != "elitesearch/videos.py"}
            if saved != current:
                raise ValueError(f"Cannot reuse this run: {key} changed")
        if upgrading_repairs:
            # The old baseline disabled repairs; it cannot represent the new matched
            # optimizer. Archive it once, including across interrupted migrations.
            migration = output / "repair-upgrade.json"
            if not migration.exists():
                save_json(migration, dict(previous=previous, status="started"))
            if read_json(migration)["status"] != "baseline_archived":
                if (output / "baseline").exists():
                    archive(output / "baseline")
                save_json(migration, dict(previous=previous, status="baseline_archived"))
            for objective in ("performance", "tokens", "evaluations"):
                upgrade_candidates(output / objective, upgrade="repairs")
            progress(
                "Enabled enforced repairs; original proposals retained, old baseline and candidate results archived"
            )
            (output / "config.yaml").write_text(
                yaml.safe_dump(config.model_dump(), sort_keys=False)
            )
        elif previous["protocol"] in ("ocean-elitetable-meta-v4", "ocean-elitetable-meta-v5"):
            for objective in ("performance", "tokens", "evaluations"):
                upgrade_candidates(output / objective, upgrade="task-context")
            progress(
                "Enabled task context for outer and inner models; reusing the EliteTable baseline "
                "and first-generation programs, archiving old candidate scores for reevaluation"
            )
        elif previous["protocol"] == "ocean-elitetable-meta-v6":
            progress(
                "Updated worker reporting; preserving all saved proposals, results and budgets"
            )
        elif (
            previous["protocol"] != manifest["protocol"]
            or previous["source_hashes"] != manifest["source_hashes"]
        ):
            raise ValueError("Resume requires matching protocol and source hashes")
        append_json(
            output / "resumes.jsonl",
            dict(
                time=datetime.now(timezone.utc).isoformat(),
                previous=previous,
                current=manifest,
                worker_settings={
                    key: dict(previous=saved_config[key], current=config.model_dump()[key])
                    for key in worker_settings
                },
            ),
        )
        (output / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
    else:
        (output / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
    save_json(output / "manifest.json", manifest)
    loop = asyncio.new_event_loop()
    summary = dict(status="running", campaigns={})
    try:
        anchors = (
            {} if elite else loop.run_until_complete(calibrate(config, output / "calibration"))
        )
        if not elite:
            summary["anchors"] = anchors
        if elite:
            # Release setup's Live display so the campaign can own the terminal leaderboard.
            progress(
                "Setup complete; proposing sub-evolvers and measuring the EliteTable baseline",
                kind="search_finished",
                status="completed",
                reason="setup completed",
            )
        objectives = (
            ["performance", "tokens", "evaluations"]
            if config.objective == "all"
            else [config.objective]
        )
        for objective in objectives:
            track = config.model_copy(update={"objective": objective})
            summary["campaigns"][objective] = run_campaign(track, output / objective, anchors, loop)
        summary["status"] = "completed"
        return summary
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Reap subprocesses even after keyboard interruption before closing the loop.
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        save_json(output / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Meta optimization of isolated multi-environment policy evolvers"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing EliteTable run; stop its old process first",
    )
    parser.add_argument("--objective", choices=("performance", "tokens", "evaluations", "all"))
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Resolve config and workload without calls or workers",
    )
    args = parser.parse_args(argv)
    values = read_config(args.config)
    if args.objective:
        values["objective"] = args.objective
    config_type = Config
    if values.get("optimizer") == "elitetable":
        from .elitetable import Config as config_type
    config = config_type.model_validate(values)
    if args.print_config:
        print(
            yaml.safe_dump(
                dict(config=config.model_dump(), workload=workload(config)), sort_keys=False
            )
        )
        return
    if args.output is None:
        parser.error("--output is required unless --print-config is used")
    run(config, args.output.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
