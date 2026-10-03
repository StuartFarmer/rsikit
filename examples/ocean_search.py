"""Budgeted Ocean 2048 EliteSearch or independent proposals (cooperative policies).

No paid calls occur until this command is explicitly run with model and price ceilings.
Reservations are conservative token-price accounting, not a provider billing guarantee.
Use a provider-side account cap if a strict external billing limit is required.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from statistics import fmean, median

from slick import prompts
from slick.providers import OpenRouterAPI

from research import elitesearch
from research.elitesearch import Config, EliteSearch
from rsikit import Policy, Run

CALL = contextvars.ContextVar("ocean_call", default={})
RESPONSE = contextvars.ContextVar("ocean_response", default=None)
CONTEXT = """Ocean g2048: deterministic, stateless Python/NumPy batch policy.
Observation is float32 (B,16), row-major 4x4 tile exponents (0=empty).
Return integer actions (B,): 0=up, 1=down, 2=left, 3=right.
Every row must depend only on its own board, with no evolving shared state,
cross-board statistics, or randomness. Scalar evaluation uses B=1.
Maximize conventional tile merge score, not maximum tile or shaped return.
Native adaptive timeout and invalid-move penalties apply; external cap is 2000
steps. No environment, file, process, network, clock or evaluator access.
"""


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


class BudgetExceeded(RuntimeError):
    pass


class UsageOpenRouter(OpenRouterAPI):
    async def _asend(self, request):
        response = await super()._asend(request)
        event = RESPONSE.get()
        usage = getattr(response, "usage", None)
        if event is not None and usage is not None:
            event["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            event["actual_cost"] = event["usage"].get("cost")
        return response


class BudgetProvider:
    """Reserve full per-call limits before sending; failed/cancelled calls are charged.

    We never refund unknown usage or assume an API error means zero cost. The UTF-8
    byte count plus 1024 framing tokens bounds ordinary byte-tokenized text requests;
    no tools/images are allowed. Price ceilings must cover the chosen model/routing.
    Actual billing remains unknown when the API omits cost metadata.
    """

    def __init__(
        self,
        provider,
        *,
        max_calls,
        max_tokens,
        spend_cap,
        max_input_tokens,
        max_output_tokens,
        input_price,
        output_price,
        log=None,
    ):
        self.provider = provider
        self.limits = dict(
            max_calls=max_calls,
            max_tokens=max_tokens,
            spend_cap=spend_cap,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            input_price=input_price,
            output_price=output_price,
        )
        for name, value in self.limits.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.per_call_tokens = max_input_tokens + max_output_tokens
        self.per_call_cost = (
            Decimal(str(input_price)) * max_input_tokens
            + Decimal(str(output_price)) * max_output_tokens
        ) / 1000000
        self.calls = 0
        self.events = []
        self.stopped = None
        self.log = log

    @property
    def reserved_tokens(self):
        return self.calls * self.per_call_tokens

    @property
    def reserved_cost(self):
        return float(self.calls * self.per_call_cost)

    async def acall(self, context, **kwargs):
        if kwargs:
            raise ValueError("Ocean generation supports text-only calls without tools")
        reason = self.stopped
        if len(context.encode("utf-8")) + 1024 > self.limits["max_input_tokens"]:
            reason = "input token bound"
        elif self.calls >= self.limits["max_calls"]:
            reason = "call cap"
        elif self.reserved_tokens + self.per_call_tokens > self.limits["max_tokens"]:
            reason = "token cap"
        elif (self.calls + 1) * self.per_call_cost > Decimal(str(self.limits["spend_cap"])):
            reason = "spend reservation cap"
        if reason:
            self.stopped = reason
            raise BudgetExceeded(reason)
        # No await before reservation: concurrent requests cannot overspend the ledger.
        self.calls += 1
        event = dict(
            CALL.get(),
            call=self.calls,
            start=time.monotonic(),
            finish=None,
            status="running",
            usage=None,
            actual_cost=None,
            reserved_tokens=self.per_call_tokens,
            reserved_cost=float(self.per_call_cost),
        )
        self.events.append(event)
        token = RESPONSE.set(event)
        if self.log:
            self.log(dict(event, event="llm_start"))
        try:
            result = await self.provider.acall(context)
            event["status"] = "ok"
            return result
        except BaseException as exc:
            event.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            event["finish"] = time.monotonic()
            RESPONSE.reset(token)
            if self.log:
                self.log(dict(event, event="llm_finish"))


class Search(EliteSearch):
    def __init__(self, *args, independent=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.independent = independent
        self.ranking_seconds = {}
        if independent and (self.config.new_fraction != 1 or self.config.remix_fraction != 0):
            raise ValueError("Independent proposals require only new proposals")

    async def invent(self, proposal, elites, **kwargs):
        return await super().invent(proposal, [] if self.independent else elites, **kwargs)

    async def _call(self, row, operation, *args):
        token = CALL.set(
            dict(
                generation=row.generation,
                attempt=row.id,
                repair=row.repairs,
                operation=operation.__name__,
            )
        )
        try:
            return await super()._call(row, operation, *args)
        finally:
            CALL.reset(token)

    async def _generate(self, row, **kwargs):
        try:
            return await super()._generate(row, **kwargs)
        except BudgetExceeded as exc:
            # Drain admitted candidates before stopping so valid arrivals are retained.
            row.status, row.error = "discarded", str(exc)
            return row

    def _promote(self, generation, rows):
        start = time.monotonic()
        super()._promote(generation, rows)
        self.ranking_seconds[generation.number] = time.monotonic() - start
        if self.provider.stopped:
            raise BudgetExceeded(self.provider.stopped)


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
        if validation[p.id].accepted and set(validation[p.id].scores) == set(range(1000, 1128))
    ]
    winner = (
        max(valid, key=lambda p: fmean(validation[p.id].scores.values())) if valid else fallback
    )
    # Persist the frozen choice before opening the test panel. No repairs after search.
    winner.to_file(run.path / "winner.py")
    save_json(
        run.path / "selection.json",
        dict(
            winner=winner.id,
            fallback=not valid,
            validation={k: asdict(v) for k, v in validation.items()},
        ),
    )
    test = (await evaluate([winner], range(2000, 2512)))[winner.id]
    return dict(
        winner=winner.id,
        fallback=not valid,
        validation={k: asdict(v) for k, v in validation.items()},
        test=asdict(test),
        test_mean=(
            fmean(test.scores.values())
            if test.accepted and set(test.scores) == set(range(2000, 2512))
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
            run.save_policy(policy, scores=result.scores if result.accepted else {})
            row = next((r for r in trace_rows if r["policy_id"] == policy.id), None)
            if row is not None:
                row["persisted"] = time.monotonic()
                job = next(
                    (e for e in reversed(evaluator.events) if e["policy_id"] == policy.id), None
                )
                if job is not None:
                    row["job_id"] = job["job_id"]
                log(dict(row, event="score_persisted", failure=result.failure))
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
            Policy.from_text(r.implementation, name=r.name, description=r.description)
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
    summary.update(await select_winner(candidates, baselines()[0], evaluate, run))
    save_json(run.path / "summary.json", summary)
    return summary


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", choices=("g2048",), default="g2048")
    parser.add_argument("--arm", choices=("elite", "independent"), default="elite")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--elites", type=int, default=5)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(32)))
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
    parser.add_argument("--batch-size", type=int, default=32)
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

    from research.ocean.evaluator import PanelEvaluator
    from research.ocean.native import FLAGS, UPSTREAM, build

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
    library = build()
    manifest = dict(
        vars(args),
        config=asdict(config),
        native_library=str(library),
        native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        upstream=UPSTREAM,
        compiler_flags=FLAGS,
        compiler=os.environ.get("CC", "cc"),
        platform=platform.platform(),
        libc=platform.libc_ver(),
        python=platform.python_version(),
        numpy=np.__version__,
        hardware_cpu_count=os.cpu_count(),
        allocated_cpus=4,
        allocated_ram_gib=8,
        resource_allocation_os_enforced=False,
        numerical_threads={name: os.environ[name] for name in thread_variables},
        build_seconds=time.monotonic() - start,
        search_seeds=args.seeds,
        validation_seeds=list(range(1000, 1128)),
        test_seeds=list(range(2000, 2512)),
        max_steps=2000,
        panel_timeout=60,
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
            max_steps=2000,
            timeout=60,
        ) as evaluator:
            agent = Search(
                f"Maximize mean tile merge score over {len(args.seeds)} seeded Ocean games.",
                provider,
                None,
                context=CONTEXT,
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
