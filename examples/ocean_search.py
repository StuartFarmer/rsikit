"""Budgeted Ocean EliteSearch or independent proposals (cooperative policies).

No paid calls occur until this command is explicitly run with model and price ceilings.
Reservations are conservative token-price accounting, not a provider billing guarantee.
Use a provider-side account cap if a strict external billing limit is required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median

from slick import prompts

from research import elitesearch
from research.elitesearch import Config
from research.elitesearch.cli import Search
from research.experiment import _evidence
from research.ocean.environment import BREAKOUT_CONTEXT, CONTEXT
from research.providers import BudgetExceeded, BudgetProvider, UsageOpenRouter
from rsikit import PolicyDefinition, Run
from rsikit.evaluation import episode_error, episode_scores


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def generation_tails(events, arrivals, checkpoints):
    result = []
    for number in sorted({e["generation"] for e in events if "generation" in e}):
        calls = [e for e in events if e.get("generation") == number]
        evaluations = [e for e in arrivals if e["generation"] == number and "persisted" in e]
        start, finish = min(e["start"] for e in calls), max(e["finish"] for e in calls)
        tail = max(0, max((e["persisted"] for e in evaluations), default=finish) - finish)
        result.append(
            dict(
                generation=number,
                llm_window=finish - start,
                residual_tail=tail,
                tail_fraction=tail / (finish - start) if finish > start else None,
                checkpoint_seconds=checkpoints.get(number, 0),
            )
        )
    return result


def active_seconds(events):
    end = total = 0.0
    for event in sorted(events, key=lambda e: e["start"]):
        total += max(0, event["finish"] - max(end, event["start"]))
        end = max(end, event["finish"])
    return total


async def select_winner(candidates, fallback, evaluate, run):
    validation = await evaluate(candidates, range(1000, 1128)) if candidates else {}
    valid = [
        p
        for p in candidates
        if (bool(validation[p.id]) and episode_error(validation[p.id]) is None)
        and set(episode_scores(validation[p.id])) == set(range(1000, 1128))
    ]
    winner = (
        max(valid, key=lambda p: fmean(episode_scores(validation[p.id]).values()))
        if valid
        else fallback
    )
    # Persist the frozen choice before opening the test panel. No repairs after search.
    winner.to_file(run.path / "winner.py")
    save_json(
        run.path / "selection.json",
        dict(
            winner=winner.id,
            fallback=not valid,
            validation={k: _evidence(v) for k, v in validation.items()},
        ),
    )
    test = (await evaluate([winner], range(2000, 2512)))[winner.id]
    return dict(
        winner=winner.id,
        fallback=not valid,
        validation={k: _evidence(v) for k, v in validation.items()},
        test=_evidence(test),
        test_mean=(
            fmean(episode_scores(test).values())
            if (bool(test) and episode_error(test) is None)
            and set(episode_scores(test)) == set(range(2000, 2512))
            else None
        ),
    )


async def run_search(agent, provider, evaluator, run, *, manifest=None):
    from research.ocean.baselines import policies as baselines

    origin, arrivals, checkpoints = time.monotonic(), [], {}
    manifest = manifest or {}
    event_path = run.path / "events.jsonl"

    def log(event):
        with event_path.open("a") as stream:
            stream.write(json.dumps(event, default=str, allow_nan=False) + "\n")

    provider.log = log

    def checkpoint(current):
        start = time.monotonic()
        run.save(*current.records())
        number = current.generations[-1].number if current.generations else 0
        checkpoints[number] = checkpoints.get(number, 0) + time.monotonic() - start

    async def evaluate(policies, seeds=range(32), *, search=False):
        trace_rows = []
        for policy in policies:
            run.save_policy(policy)
            if search:
                organism = next(r for r in reversed(agent.organisms) if r.policy_id == policy.id)
                matching = [e for e in provider.events if e.get("attempt") == organism.id]
                latest = matching[-1] if matching else {}
                source = run.path / "search_sources" / f"{policy.id}.py"
                source.parent.mkdir(exist_ok=True)
                policy.to_file(source)
                row = dict(
                    at=time.monotonic() - origin,
                    policy=str(source.relative_to(run.path)),
                    policy_id=policy.id,
                    generation=organism.generation,
                    attempt=organism.id,
                    repair=organism.repairs,
                    llm_start=latest.get("start", origin) - origin,
                    llm_finish=latest.get("finish", origin) - origin,
                )
                arrivals.append(row)
                trace_rows.append(row)
                log(dict(row, event="evaluation_submitted"))
        measured = await evaluator.evaluate(policies, seeds)
        for policy in policies:
            result = measured[policy.id]
            run.save_policy(
                policy,
                scores=episode_scores(result)
                if (bool(result) and episode_error(result) is None)
                else {},
            )
            row = next((r for r in trace_rows if r["policy_id"] == policy.id), None)
            if row is not None:
                row["persisted"] = time.monotonic()
                job = next(
                    (e for e in reversed(evaluator.events) if e["policy_id"] == policy.id), None
                )
                if job is not None:
                    row["job_id"] = job["job_id"]
                log(dict(row, event="score_persisted", failure=episode_error(result)))
        return measured

    async def search_evaluate(policies):
        return await evaluate(policies, manifest.get("search_seeds", range(32)), search=True)

    agent.evaluate, agent.on_checkpoint = search_evaluate, checkpoint
    error = None
    try:
        await agent.run()
    except BudgetExceeded:
        agent.reason = "budget_exhausted"
        agent.generations[-1].status = "completed"
        agent.generations[-1].error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # Also retain completed rows from an interrupted generation, before promotion.
        ranked = sorted(
            (r for r in agent.organisms if r.score is not None), key=lambda r: (-r.score, r.id)
        )[:5]
        candidates = [
            PolicyDefinition.from_text(r.implementation, name=r.name, description=r.description)
            for r in ranked
        ]
        checkpoint(agent)
        tails = generation_tails(provider.events, arrivals, checkpoints)
        for tail in tails:
            tail["ranking_seconds"] = agent.ranking_seconds.get(tail["generation"], 0)
        save_json(run.path / "generation_tails.json", tails)
        save_json(
            run.path / "arrival_trace.json",
            dict(
                arrivals=[{k: v for k, v in row.items() if k != "persisted"} for row in arrivals],
                model=manifest.get("model", "scripted"),
                generation_concurrency=agent.config.generation_concurrency,
                provider_limits=provider.limits,
                active_generation_seconds=active_seconds(provider.events),
                duration_seconds=time.monotonic() - origin,
                llm_latencies=[e["finish"] - e["start"] for e in provider.events],
            ),
        )
        summary = dict(
            reason=agent.reason,
            error=error,
            calls=provider.calls,
            reserved_tokens=provider.reserved_tokens,
            reserved_cost=provider.reserved_cost,
            actual_cost=(
                sum(e["actual_cost"] for e in provider.events)
                if all(e["actual_cost"] is not None for e in provider.events)
                else None
            ),
            proposals=len({e["attempt"] for e in provider.events}),
            repairs=sum(e["repair"] > 0 for e in provider.events),
            failed_calls=sum(e["status"] != "ok" for e in provider.events),
            failed_proposals=sum(
                r.score is None and any(e["attempt"] == r.id for e in provider.events)
                for r in agent.organisms
            ),
            budget_stop=provider.stopped,
            median_tail=median([t["residual_tail"] for t in tails]) if tails else None,
            max_tail=max([t["residual_tail"] for t in tails], default=None),
            median_tail_fraction=median(
                [t["tail_fraction"] for t in tails if t["tail_fraction"] is not None]
            )
            if tails
            else None,
        )
        save_json(run.path / "summary.json", summary)
    summary.update(
        await select_winner(candidates, baselines(manifest.get("env", "g2048"))[0], evaluate, run)
    )
    save_json(run.path / "summary.json", summary)
    return summary


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", choices=("g2048", "breakout"), default="g2048")
    parser.add_argument("--arm", choices=("elite", "independent"), default="elite")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--elites", type=int, default=5)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(32)),
        help="Independent episode seeds",
    )
    parser.add_argument("--spend-cap", type=float, required=True)
    parser.add_argument(
        "--input-price", type=float, required=True, help="Upper USD per million input tokens"
    )
    parser.add_argument(
        "--output-price", type=float, required=True, help="Upper USD per million output tokens"
    )
    parser.add_argument(
        "--max-calls", type=int, help="Default: population × generations × (1 + max-repairs)"
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Reservation ceiling; default: max-calls × per-call limits"
    )
    parser.add_argument("--max-input-tokens", type=int, default=65536)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--generation-concurrency", type=int, default=4)
    parser.add_argument("--generation-timeout", type=float, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="Panel timeout allowance in seconds per episode seed",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Maximum concurrent episodes")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps per episode")
    parser.add_argument(
        "--score-key", choices=("return", "merge_score", "score", "episode_return", "perf")
    )
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--mode", choices=("reference", "summary", "batch"), default="batch")
    parser.add_argument("--search-seed", type=int, default=0)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if (
            isinstance(value, (int, float))
            and key not in ("search_seed", "max_repairs")
            and (not math.isfinite(value) or value <= 0)
        ):
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.max_repairs < 0:
        parser.error("--max-repairs must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("--seeds must contain unique unsigned 32-bit integers")
    if not isinstance(args.env_kwargs, dict):
        parser.error("--env-kwargs must be a JSON object")
    if any(1000 <= s < 1128 or 2000 <= s < 2512 for s in args.seeds):
        parser.error("Search seeds must exclude validation 1000–1127 and test 2000–2511")
    if args.max_calls is None:
        args.max_calls = args.population * args.generations * (1 + args.max_repairs)
    if args.max_tokens is None:
        args.max_tokens = args.max_calls * (args.max_input_tokens + args.max_output_tokens)
    if args.workers > 4:
        parser.error("--workers must fit the frozen four-core allocation")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before launching paid generation")
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for name in thread_variables:
        os.environ[name] = "1"  # Spawned workers inherit these before importing NumPy.
    import numpy as np

    from research.ocean.environment import metadata
    from research.ocean.evaluator import PanelEvaluator, _inputs

    try:
        _inputs(args.seeds, args.batch_size, args.max_steps)
    except ValueError as exc:
        parser.error(str(exc))
    args.score_key = args.score_key or ("merge_score" if args.env == "g2048" else "return")

    prompts.TEMPLATE_ROOT = Path(elitesearch.__file__).parent / "prompts"
    raw = UsageOpenRouter(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        timeout=args.generation_timeout,
        max_retries=0,
    )
    provider = BudgetProvider(
        raw,
        **{
            k: getattr(args, k)
            for k in (
                "max_calls",
                "max_tokens",
                "spend_cap",
                "max_input_tokens",
                "max_output_tokens",
                "input_price",
                "output_price",
            )
        },
    )
    config = Config(
        elite_size=args.elites,
        population_size=args.population,
        generations=args.generations,
        max_repairs=args.max_repairs,
        new_fraction=1 if args.arm == "independent" else 0.2,
        remix_fraction=0 if args.arm == "independent" else 0.4,
        generation_concurrency=args.generation_concurrency,
        generation_timeout=args.generation_timeout,
    )
    start = time.monotonic()
    upstream = metadata(args.env)
    manifest = dict(
        vars(args),
        optimization_schedule="round-v1",
        config=asdict(config),
        **upstream,
        platform=platform.platform(),
        libc=platform.libc_ver(),
        python=platform.python_version(),
        numpy=np.__version__,
        hardware_cpu_count=os.cpu_count(),
        allocated_cpus=4,
        allocated_ram_gib=8,
        resource_allocation_os_enforced=False,
        numerical_threads={name: os.environ[name] for name in thread_variables},
        import_seconds=time.monotonic() - start,
        search_seeds=args.seeds,
        validation_seeds=list(range(1000, 1128)),
        test_seeds=list(range(2000, 2512)),
        max_transitions_per_candidate=len(args.seeds) * args.max_steps,
        timeout_per_seed=args.timeout,
        cooperative_execution=True,
        budget_accounting="Unrefunded worst-case token-price reservations; actual billing may be unavailable",
        input_bound="UTF-8 bytes + 1024 framing tokens; byte-tokenized text models only",
    )
    async with Run.create(name=f"ocean-{args.arm}", path=args.output) as run:
        save_json(run.path / "manifest.json", manifest)
        async with PanelEvaluator(
            run.path / "panels",
            mode=args.mode,
            workers=args.workers,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            env_name=args.env,
            score_key=args.score_key,
            env_kwargs=args.env_kwargs,
            timeout=args.timeout,
        ) as evaluator:
            agent = Search(
                f"Maximize {args.score_key} averaged over {len(args.seeds)} upstream Ocean episode seeds; "
                f"one game per seed, up to {args.max_steps} steps per episode. "
                "Score comes from the final state, including capped episodes; "
                "return is cumulative episode reward.",
                provider,
                None,
                context=(CONTEXT if args.env == "g2048" else BREAKOUT_CONTEXT)
                + f"\nConfigured environment kwargs: {json.dumps(args.env_kwargs)}\n",
                config=config,
                seed=args.search_seed,
                independent=args.arm == "independent",
            )
            summary = await run_search(agent, provider, evaluator, run, manifest=manifest)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("validation", "test")}, indent=2)
    )


if __name__ == "__main__":
    asyncio.run(main())
