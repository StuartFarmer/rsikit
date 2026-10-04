"""EliteTable evolves executable sub-evolvers; the host owns generations and budgets."""

import asyncio
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from time import monotonic
from typing import Literal

from pydantic import Field, ValidationError, model_validator
from rich.console import Console
from rich.table import Column
from slick import prompts
from slick.providers import ProviderError

from research.elitesearch.agent import Config as SearchConfig
from research.elitesearch.cli import Search
from research.elitesearch.healing import SelfHealer
from research.elitesearch.records import Generation, Organism
from research.experiment import EvaluationConfig
from research.ocean.baselines import policies
from research.ocean.environment import definition, describe
from research.providers import BudgetExceeded, BudgetProvider, UsageOpenRouter
from research.rewards import Measurement
from rsikit.evaluation import PolicyError
from rsikit.policy import InvalidPolicy
from rsikit.progress import bind_run

from .checkpoints import archive, read_events, read_json, save_json
from .experiment import Config as BaseConfig
from .experiment import Trial, append_json, digest, progress, source_text, usage
from .sandbox import Worker

CONTRACT = """Write a sub-evolver, not a game-playing policy. Define Solution(Policy),
importing Policy from rsikit, inheriting its constructor. Implement async act(self, observation).
The trusted host invokes act exactly five times, once per generation. Return a nonempty
list of complete game-policy Python source strings, no more than observation['generation_size']
(hard maximum 50). The host evaluates every submitted policy on ten seeds in the current task.
Each original proposal consumes one population slot. The host enforces up to five repairs
of every invalid or crashing policy, using its exact source and diagnostic, before returning
the slot's final result. Repairs are revisions in that slot, not additional population members.
Every repair model call and reevaluation is charged against the trusted trial budgets.
observation contains: task (game instructions), starter (valid game policy source), seed,
generation (1..5), generations (5), generation_size, and results from the previous generation.
Each result includes source plus id/score/scores, or error.
Retain useful history in self, initialized in async reset.
Feedback source is the final repaired source when repair succeeds. Do not assume it is
identical to the submitted source. You cannot disable or bypass host-enforced repair.
Use await self.generate(list_of_prompts) to obtain ordered responses: {'text': source} or
{'error': diagnostic}. At most generation_size model calls are allowed each generation.
Requests in that list run concurrently. Each request should ask for one complete game policy.
The host includes the current task specification in every generation request automatically;
use your prompts for search strategy, parent source and score feedback.
Do not call self.generate during reset. No other oracles, network, files, processes,
simulators or credentials. Do not create an environment or execute proposed game policies.
All generation/model usage is metered. Each act call may choose any population size 1..cap.
Choose parents, prompts, mutations, crossover and diversity in YOUR program.
The host retains the highest search-scoring valid policy across all five generations and
privately audits it after your container exits. Private audit scores are never exposed to you.
The same sub-evolver is tested from scratch across environments and replicates.
"""


class Config(BaseConfig):
    optimizer: Literal["elitetable"] = "elitetable"
    objective: Literal["performance", "tokens", "evaluations", "all"] = "performance"
    population: int = Field(default=10, ge=1)
    generations: int = Field(default=5, ge=1)
    elites: int = Field(default=10, ge=1)
    max_repairs: int = Field(default=5, ge=0)
    inner_max_repairs: int = Field(default=5, ge=0, le=5)
    inner_generations: Literal[5] = 5
    generation_size: int = Field(default=50, ge=1, le=50)
    trial_workers: int = Field(default=8, ge=1)
    evaluation_workers: int = Field(default=16, ge=1)
    model_workers: int = Field(default=8, ge=1)
    output_tokens: int = Field(default=1_024_000, ge=1)
    trial_spend_cap: float = Field(default=25, gt=0)
    editor_spend_cap: float = Field(default=25, gt=0)
    # GEPA-only legacy fields are absent from the EliteTable config snapshot.
    benchmark_limit: int = Field(default=9, exclude=True)
    revisions: int = Field(default=2, exclude=True)
    calibration_cases: int = Field(default=64, exclude=True)

    @model_validator(mode="after")
    def generation_limits(self):
        if self.elites > self.population:
            raise ValueError("elites must not exceed outer population")
        self.evaluations = (
            self.inner_generations * self.generation_size * (1 + self.inner_max_repairs)
        )
        return self


def provider(config, path, *, editor=False):
    calls = (
        config.population * config.generations * (1 + config.max_repairs)
        if editor
        else config.evaluations
    )
    cap = config.editor_output_tokens if editor else config.max_output_tokens
    if not editor:
        calls = min(calls, config.output_tokens // cap)

    def log(event):
        append_json(path, event)
        progress(
            f"{'Meta optimizer' if editor else 'Sub-evolver'} model call {event['call']}/{calls}: "
            f"{event['status']}; cost={event['actual_cost']}"
        )

    model = BudgetProvider(
        UsageOpenRouter(
            model=config.model,
            max_output_tokens=cap,
            timeout=config.generation_timeout,
            max_retries=0,
        ),
        max_calls=calls,
        max_tokens=calls * (config.max_input_tokens + cap),
        spend_cap=config.editor_spend_cap if editor else config.trial_spend_cap,
        max_input_tokens=config.max_input_tokens,
        max_output_tokens=cap,
        input_price=config.input_price,
        output_price=config.output_price,
        log=log,
    )
    model.restore(read_events(path))
    return model


def task_context(config, environment):
    task = definition(environment)
    return describe(
        environment,
        task.Options(),
        EvaluationConfig(
            max_steps=config.max_steps,
            batch_size=config.batch_size,
            score_key=task.evaluation_defaults["score_key"],
        ),
    )


class WorkerPool(asyncio.Semaphore):
    """The shared slot limit plus counters for the campaign's Rich display."""

    def __init__(self, limit):
        super().__init__(limit)
        self.limit = limit
        self.active = self.queued = self.finished = 0

    async def acquire(self):
        self.queued += 1
        try:
            result = await super().acquire()
        finally:
            self.queued -= 1
        self.active += 1
        return result

    def release(self):
        self.active -= 1
        self.finished += 1
        super().release()


async def report_workers(slots):
    previous = None
    while True:
        pools = {
            name: dict(
                active=pool.active, limit=pool.limit, queued=pool.queued, finished=pool.finished
            )
            for name, pool in (
                ("Searches", slots["trials"]),
                ("Panels", slots["evaluations"]),
                ("Models", slots["models"]),
            )
        }
        if pools != previous:
            progress(
                "Workers: "
                + "; ".join(
                    f"{name} {p['active']}/{p['limit']}, queued {p['queued']}, finished {p['finished']}"
                    for name, p in pools.items()
                ),
                kind="workers",
                pools=pools,
            )
            previous = pools
        await asyncio.sleep(1)


class EvolverTrial(Trial):
    def __init__(self, *args, model_slots, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_slots = model_slots
        self.generations = self.generation_calls = 0
        self.best = float("-inf")
        self.stop_reason = self.stop_detail = None
        self.task = "Implement a valid game-playing policy for the supplied environment."

    def stop_for_budget(self, reason):
        self.stop_reason, self.stop_detail = "budget_exhausted", str(reason)
        progress(f"{self.path}: budget exhausted ({reason}); auditing the best policy found")

    async def evaluate_with_repairs(self, source, slot):
        """A population slot cannot bypass the same repair agent used by EliteTable."""
        submitted = source
        response = await super().handle(dict(op="evaluate", value=source))
        repairs = 0
        healer = SelfHealer(self.task, self.model)
        while "error" in response and repairs < self.config.inner_max_repairs:
            if response["error"].startswith("BudgetExceeded:"):
                break
            repairs += 1
            record = dict(
                generation=self.generations,
                slot=slot,
                attempt=repairs,
                failed_source=source,
                diagnostic=response["error"],
                status="running",
            )
            append_json(self.path / "repairs.jsonl", record)
            progress(
                f"{self.path}: generation {self.generations}, slot {slot}: repairing {repairs}/{self.config.inner_max_repairs}: {response['error']}"
            )
            call = {}
            try:
                async with self.model_slots:
                    repaired = await asyncio.wait_for(
                        healer.repair("", str(source), response["error"], record=call),
                        self.config.generation_timeout,
                    )
                source = source_text(repaired._implementation)
                response = await super().handle(dict(op="evaluate", value=source))
            except BudgetExceeded as exc:
                response = dict(error=f"BudgetExceeded: repair stopped: {exc}")
            except (
                ValidationError,
                InvalidPolicy,
                ValueError,
                ProviderError,
                asyncio.TimeoutError,
            ) as exc:
                source = call.get("raw", source)
                response = dict(error=f"{type(exc).__name__}: {exc}")
            except asyncio.CancelledError:
                record["status"] = "cancelled"
                response = dict(error="CancelledError: repair interrupted")
                raise
            finally:
                append_json(
                    self.path / "repairs.jsonl",
                    dict(
                        record,
                        status="cancelled" if record["status"] == "cancelled" else "finished",
                        call=call,
                        result=response,
                        source=source,
                    ),
                )
        if "id" in response:
            source = self.accepted[response["id"]]
            if repairs:
                progress(
                    f"{self.path}: generation {self.generations}, slot {slot}: repaired successfully after {repairs} attempt(s)"
                )
        elif repairs:
            progress(
                f"{self.path}: generation {self.generations}, slot {slot}: repair failed after {repairs} attempt(s): {response['error']}"
            )
        if response.get("error", "").startswith("BudgetExceeded:"):
            self.stop_for_budget(response["error"])
        return dict(source=source, submitted_source=submitted, repairs=repairs, **response)

    async def handle(self, request):
        op = request.get("op") if isinstance(request, dict) else None
        value = request.get("value") if isinstance(request, dict) else None
        try:
            if self.generations >= self.config.inner_generations:
                raise BudgetExceeded("five-generation cap")
            if op == "generate":
                if (
                    not isinstance(value, list)
                    or not value
                    or any(not isinstance(p, str) for p in value)
                ):
                    raise ValueError("generate requires a nonempty list of prompts")
                if self.generation_calls + len(value) > self.config.generation_size:
                    raise BudgetExceeded("per-generation model-call cap")
                self.generation_calls += len(value)

                async def generate(prompt):
                    async with self.model_slots:
                        response = await super(EvolverTrial, self).handle(
                            dict(
                                op="generate",
                                value=f"Current game task (provided by the host):\n{self.task}"
                                f"\n\nSub-evolver request:\n{prompt}",
                            )
                        )
                        if "text" in response:
                            response["text"] = source_text(response["text"])
                        elif response.get("error", "").startswith("BudgetExceeded:"):
                            self.stop_for_budget(response["error"])
                        return response

                return dict(responses=await asyncio.gather(*(generate(p) for p in value)))
            if op != "population":
                raise ValueError("Sub-evolvers may only generate and submit populations")
            if (
                not isinstance(value, dict)
                or type(value.get("generation")) is not int
                or value["generation"] != self.generations + 1
            ):
                raise ValueError("Generations must be submitted once, in order, from 1 to 5")
            population = value.get("policies")
            if (
                not isinstance(population, list)
                or not 1 <= len(population) <= self.config.generation_size
            ):
                raise ValueError(
                    f"Population must contain 1..{self.config.generation_size} policies"
                )
            # Admission precedes evaluation; invalid sources also consume their slots.
            self.generations += 1
            progress(
                f"{self.path}: inner generation {self.generations}/5, {len(population)} policies"
            )
            results = await asyncio.gather(
                *(
                    self.evaluate_with_repairs(source, slot)
                    for slot, source in enumerate(population, 1)
                )
            )
            for response in results:
                if "id" in response and response["score"] > self.best:
                    self.best = response["score"]
                    await super().handle(dict(op="commit", value=response["id"]))
            save_json(self.path / f"generation-{self.generations}.json", results)
            self.generation_calls = 0
            return dict(results=results)
        except Exception as exc:
            response = dict(error=f"{type(exc).__name__}: {exc}")
            append_json(self.path / "oracle.jsonl", dict(request=request, response=response))
            return response


def comparison_scorecard(rows, baseline, objective="performance"):
    """Bounded symmetric percentage improvement; equal weight for each environment."""
    keys = [(r["environment"], r["replicate"]) for r in rows]
    reference_keys = [(r["environment"], r["replicate"]) for r in baseline]
    if (
        not rows
        or len(set(keys)) != len(keys)
        or len(set(reference_keys)) != len(reference_keys)
        or set(keys) != set(reference_keys)
    ):
        raise ValueError(
            "Candidate and EliteTable baseline must have identical unique environment/replicate panels"
        )
    if any(
        r.get("error") or r.get("audit_error") or not math.isfinite(r["score"]) for r in baseline
    ):
        raise ValueError("EliteTable baseline must complete successfully before comparison")

    def mean_tokens(panel):
        return (
            None
            if any(r["output_tokens"] is None for r in panel)
            else fmean(r["output_tokens"] for r in panel)
        )

    environments = {}
    for name in sorted({key[0] for key in keys}):
        candidate = [r for r in rows if r["environment"] == name]
        reference = [r for r in baseline if r["environment"] == name]
        score, base = fmean(r["score"] for r in candidate), fmean(r["score"] for r in reference)
        denominator = abs(score) + abs(base)
        environments[name] = dict(
            S=200 * (score - base) / denominator if denominator else 0,
            candidate_score=score,
            baseline_score=base,
            delta=score - base,
            T=mean_tokens(candidate),
            E=fmean(r["evaluations"] for r in candidate),
            baseline_T=mean_tokens(reference),
            baseline_E=fmean(r["evaluations"] for r in reference),
        )
    score = fmean(r["S"] for r in environments.values())
    tokens = (
        None
        if any(r["T"] is None for r in environments.values())
        else fmean(r["T"] for r in environments.values())
    )
    evaluations = fmean(r["E"] for r in environments.values())
    token_ratio = 1_000_000 * score / tokens if tokens else None
    evaluation_ratio = 100 * score / evaluations if evaluations else None
    ratio = token_ratio if objective == "tokens" else evaluation_ratio
    selection = (
        score
        if objective == "performance"
        else -1e30
        if ratio is None
        else score
        if score < 0
        else 1 + ratio / (1 + ratio)
    )
    return dict(
        S=score,
        T=tokens,
        E=evaluations,
        qualified=score >= 0,
        token_efficiency=token_ratio,
        evaluation_efficiency=evaluation_ratio,
        selection_score=selection,
        environments=environments,
        baseline="fixed_elitetable",
        normalization="symmetric_percentage_gain",
    )


class FixedBaseline(Search):
    """The existing engine, including its enforced policy repair loop."""

    def __init__(self, *args, trial, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial

    async def _generate(self, row, **kwargs):
        revisions = len(row.revisions)
        result = await super()._generate(row, **kwargs)
        errors = [r["error"] for r in row.revisions[revisions:] if r["status"] == "rejected"]
        if result.status == "discarded" and result.policy_id is None:
            errors.append(result.error)
        for error in errors:
            # Runtime failures are charged by Trial.handle; pre-evaluation rejections
            # must also count, including failed revisions that were eventually repaired.
            self.trial.evaluations += 1
            self.trial.failures += 1
            append_json(
                self.trial.path / "oracle.jsonl",
                dict(event="baseline_proposal_rejected", generation=row.generation, error=error),
            )
        return result


async def execute_baseline(config, trial, context, slots):
    async def evaluate(candidates):
        async def measure(policy):
            response = await Trial.handle(trial, dict(op="evaluate", value=policy._implementation))
            if "error" in response:
                return policy.id, Measurement(failure=response["error"])
            if response["score"] > trial.best:
                trial.best = response["score"]
                await Trial.handle(trial, dict(op="commit", value=response["id"]))
            return policy.id, Measurement(dict(zip(trial.seeds, response["scores"])))

        return dict(await asyncio.gather(*(measure(p) for p in candidates)))

    def checkpoint(agent):
        trial.generations = sum(g.status == "completed" for g in agent.generations)
        save_json(
            trial.path / "baseline_checkpoint.json",
            dict(
                organisms=[r.model_dump() for r in agent.organisms],
                generations=[r.model_dump() for r in agent.generations],
            ),
        )
        save_json(trial.path / "baseline_organisms.json", [r.model_dump() for r in agent.organisms])
        save_json(
            trial.path / "baseline_generations.json", [r.model_dump() for r in agent.generations]
        )

    settings = SearchConfig(
        population_size=config.generation_size,
        generations=5,
        elite_size=min(10, config.generation_size),
        max_repairs=config.inner_max_repairs,
        generation_concurrency=config.model_workers,
        generation_timeout=config.generation_timeout,
    )
    save_json(trial.path / "baseline_config.json", asdict(settings))
    agent = FixedBaseline(
        context["task"],
        trial.model,
        evaluate,
        context=context["task"] + "\nCommon available starter policy:\n" + context["starter"],
        config=settings,
        seed=context["seed"],
        trial=trial,
        on_checkpoint=checkpoint,
    )
    agent.libraries = (
        "Worker libraries: Python standard library, NumPy, Gymnasium, and rsikit.Policy."
    )
    agent._call_slots = slots["models"]
    checkpoint_path = trial.path / "baseline_checkpoint.json"
    if checkpoint_path.exists():
        saved = read_json(checkpoint_path)
        agent.restore(
            [Organism.model_validate(r) for r in saved["organisms"]],
            [Generation.model_validate(r) for r in saved["generations"]],
        )
    elif (trial.path / "baseline_generations.json").exists():
        agent.restore(
            [Organism.model_validate(r) for r in read_json(trial.path / "baseline_organisms.json")],
            [
                Generation.model_validate(r)
                for r in read_json(trial.path / "baseline_generations.json")
            ],
        )
    # Baseline engine events have their own log and cannot replace the meta leaderboard.
    with open(os.devnull, "w") as sink, bind_run(trial.path, console=Console(file=sink)):
        try:
            await agent.run()
        except BudgetExceeded as exc:
            trial.stop_for_budget(exc)
            return
    if trial.generations != 5:
        raise PolicyError("EliteTable baseline did not complete five generations")


def restore_trial(trial):
    events = read_events(trial.path / "oracle.jsonl")
    trial.evaluations = sum(
        e.get("event") in ("evaluation_admitted", "baseline_proposal_rejected") for e in events
    )
    successful = 0
    for event in events:
        request, response = event.get("request", {}), event.get("response", {})
        if request.get("op") == "evaluate" and "id" in response:
            successful += 1
            source = source_text(request["value"])
            trial.accepted[response["id"]] = source
            if response["score"] > trial.best:
                trial.best, trial.incumbent = response["score"], source
    trial.failures = trial.evaluations - successful
    trial.transitions = sum(read_json(p)["steps"] for p in trial.path.glob("panel-*.json"))
    if (trial.path / "restart.json").exists():
        trial.transitions += read_json(trial.path / "restart.json")["completed_transitions"]


async def run_trial(config, source, path, replicate, environment, slots, *, model=None):
    from .runner import evaluate_policy

    async with slots["trials"]:
        started = monotonic()
        if path.exists() and source is not None:
            if (path / "evolver.py").read_text() != source:
                raise ValueError(f"Saved sub-evolver source mismatch: {path}")
        legacy_budget_error = (
            "PolicyError: EliteTable baseline stopped before five complete generations: "
        )
        if (path / "summary.json").exists():
            row = read_json(path / "summary.json")
            if row["environment"] != environment or row["replicate"] != replicate:
                raise ValueError(f"Saved trial panel mismatch: {path}")
            if source is None and (row.get("error") or "").startswith(legacy_budget_error):
                archive(path / "summary.json")
                progress(
                    f"Recovering budget-limited baseline; only its private audit remains: {path}"
                )
            else:
                progress(f"Reusing completed trial: {path}")
                return row
        if source is not None and path.exists() and not (path / "commit.json").exists():
            # Arbitrary in-container Python state cannot be restored safely. Restart only
            # this trial, retaining all prior charges against the original trial limits.
            previous = archive(path)
            path.mkdir(parents=True)
            for name in ("model_calls.jsonl", "oracle.jsonl"):
                if (previous / name).exists():
                    shutil.copy2(previous / name, path / name)
            transitions = sum(read_json(p)["steps"] for p in previous.glob("panel-*.json"))
            if (previous / "restart.json").exists():
                transitions += read_json(previous / "restart.json")["completed_transitions"]
            save_json(
                path / "restart.json",
                dict(previous=str(previous), completed_transitions=transitions),
            )
            progress(f"Restarting interrupted sub-evolver with its remaining budget: {path}")
        path.mkdir(parents=True, exist_ok=True)
        if source is not None:
            (path / "evolver.py").write_text(source)
        model = model if model is not None else provider(config, path / "model_calls.jsonl")
        starter = policies(environment)[0]._implementation
        count = max((int(p.stem.split("-")[1]) for p in path.glob("panel-*.json")), default=0)

        async def evaluate(candidate, seeds):
            nonlocal count
            count += 1
            panel = path / f"panel-{count}.json"
            async with slots["evaluations"]:
                return await evaluate_policy(
                    config,
                    candidate,
                    seeds,
                    panel,
                    environment=environment,
                    show_progress=False,
                    label_prefix="baseline" if source is None else digest(source)[:8],
                )

        trial = EvolverTrial(
            config, path, starter, evaluate, model, replicate, model_slots=slots["models"]
        )
        restore_trial(trial)
        context = dict(
            task=task_context(config, environment),
            starter=starter,
            seed=replicate,
            generations=5,
            generation_size=config.generation_size,
        )
        if source is not None:
            context["task"] += (
                "\nThe trusted host repairs invalid policies before returning population feedback."
            )
        trial.task = context["task"]
        error = None

        async def execute():
            if source is None:
                return await execute_baseline(config, trial, context, slots)
            async with Worker(config.image, max_message=8 * 1024 * 1024) as worker:
                await worker.send(dict(mode="evolver", source=source, context=context))
                for _ in range(config.evaluations + config.inner_generations + 16):
                    request = await worker.receive()
                    if request.get("op") == "done":
                        if trial.generations != 5:
                            raise PolicyError("Sub-evolver did not complete all five generations")
                        return
                    response = await trial.handle(request)
                    await worker.send(response)
                    if request.get("op") == "population" and trial.stop_reason:
                        return
                raise PolicyError("Sub-evolver exceeded protocol request limit")

        if (path / "commit.json").exists():
            row = read_json(path / "commit.json")
            trial.incumbent = (path / "incumbent.py").read_text()
            if digest(trial.incumbent) != row["policy_hash"]:
                raise ValueError(f"Committed policy hash mismatch: {path}")
            error = row["error"]
            if source is None and (error or "").startswith(legacy_budget_error):
                archive(path / "commit.json")
                row.update(error=None, stop_reason="budget_exhausted", stop_detail=error)
                save_json(path / "commit.json", row)
                error = None
            progress(f"Resuming private audit of committed policy: {path}")
        else:
            try:
                await asyncio.wait_for(execute(), config.trial_timeout)
            except (
                PolicyError,
                asyncio.TimeoutError,
                BrokenPipeError,
                ConnectionResetError,
            ) as exc:
                if trial.stop_reason != "budget_exhausted":
                    error = f"{type(exc).__name__}: {exc}"
            # Cancellation leaves an unfinished trial, never a successful commit.
            row = trial.finish(error)
            row.update(
                environment=environment,
                replicate=replicate,
                generations=trial.generations,
                search_seconds=monotonic() - started,
                policy_hash=digest(trial.incumbent),
                stop_reason=trial.stop_reason or ("failed" if error else "completed"),
                stop_detail=trial.stop_detail,
            )
            save_json(path / "commit.json", row)
        if error is None:
            try:
                async with slots["evaluations"]:
                    audit = (
                        read_json(path / "audit.json")
                        if (path / "audit.json").exists()
                        else await evaluate_policy(
                            config,
                            trial.incumbent,
                            list(
                                range(
                                    replicate * 10000 + 1000,
                                    replicate * 10000 + 1000 + config.audit_cases,
                                )
                            ),
                            path / "audit.json",
                            environment=environment,
                            show_progress=False,
                            label_prefix="baseline" if source is None else digest(source)[:8],
                        )
                    )
                audit_score = audit.get("score")
                if audit_score is None:
                    audit_score = fmean(r["score"] for r in audit["results"])
                row.update(score=audit_score, audit_transitions=audit["steps"], audit_error=None)
            except (PolicyError, asyncio.TimeoutError, ValueError) as exc:
                row.update(
                    score=0, audit_transitions=None, audit_error=f"{type(exc).__name__}: {exc}"
                )
        else:
            row.update(score=0, audit_transitions=None, audit_error="incomplete sub-evolver")
        row["total_seconds"] = monotonic() - started
        save_json(path / "summary.json", row)
        return row


async def benchmark(config, source, path, replicates, baseline, slots):
    rows = await asyncio.gather(
        *(
            run_trial(
                config, source, path / environment / str(replicate), replicate, environment, slots
            )
            for environment in config.environments
            for replicate in replicates
        )
    )
    if source is None:
        result = dict(trials=rows, baseline="fixed_elitetable")
    else:
        result = comparison_scorecard(rows, (await baseline)["trials"], config.objective)
        result.update(trials=rows, controller_hash=digest(source))
    save_json(path / "scorecard.json", result)
    return result


class MetaSearch(Search):
    optimizer_name = "EliteTable meta"
    output_contract = CONTRACT
    libraries = "Worker libraries: Python standard library, NumPy, Gymnasium, and rsikit.Policy."
    leaderboard_columns = {
        **Search.leaderboard_columns,
        "S": Column("Gain %"),
        "T": Column("Tokens"),
        "E": Column("Evals"),
        **{name: Column(name + " Δ%") for name in ("g2048", "breakout", "maze")},
    }

    def __init__(self, *args, scorecards, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorecards = scorecards
        self.healer.output_contract = CONTRACT

    def _log_leaderboard(self):
        rows = []
        for row in self.elites:
            result = self.scorecards[row.policy_id]
            extras = dict(operation=row.kind, parents=", ".join(map(str, row.parent_ids)) or "—")
            extras.update(
                {
                    key: "unknown" if result[key] is None else f"{result[key]:.1f}"
                    for key in ("S", "T", "E")
                }
            )
            extras.update(
                {name: f"{card['S']:.1f}" for name, card in result["environments"].items()}
            )
            rows.append(
                dict(
                    id=row.policy_id,
                    name=row.name,
                    description=row.description,
                    score=row.score,
                    generation=row.generation,
                    extras=extras,
                )
            )
        progress(f"Sub-evolver leaderboard: {len(rows)} elites", kind="leaderboard", rows=rows)


async def _campaign(config, path, baselines, *, baseline_root=None):
    editor = provider(config, path / "editor_calls.jsonl", editor=True)
    slots = dict(
        trials=WorkerPool(config.trial_workers),
        evaluations=WorkerPool(config.evaluation_workers),
        models=WorkerPool(config.model_workers),
    )
    scorecards = {}
    baseline_root = baseline_root or path / "baseline"
    baseline_tasks = {}

    async def reference(phase, phase_config):
        if phase not in baselines:
            progress(
                f"Measuring fixed EliteTable baseline: {phase}; five generations per environment/replicate"
            )
            result = await benchmark(
                phase_config, None, baseline_root / phase, getattr(config, phase), None, slots
            )
            # Never compare against a broken or partially audited baseline.
            comparison_scorecard(result["trials"], result["trials"])
            baselines[phase] = result
            progress(f"Fixed EliteTable baseline ready: {phase}")
        return baselines[phase]

    def baseline_for(phase, phase_config):
        if phase not in baseline_tasks:
            baseline_tasks[phase] = asyncio.create_task(reference(phase, phase_config))
        return baseline_tasks[phase]

    async def evaluate(candidates):
        async def measure(policy):
            # A repaired source has a new policy ID. Reuse the same attempt on restart.
            directory = path / "development" / policy.id / "1"
            result = await benchmark(
                config,
                policy._implementation,
                directory,
                config.development,
                baseline_for("development", config),
                slots,
            )
            scorecards[policy.id] = result
            failure = next((r["error"] for r in result["trials"] if r["error"]), None)
            return policy.id, Measurement({0: result["selection_score"]}, failure=failure)

        return dict(await asyncio.gather(*(measure(p) for p in candidates)))

    def checkpoint(agent):
        save_json(
            path / "checkpoint.json",
            dict(
                organisms=[r.model_dump() for r in agent.organisms],
                generations=[r.model_dump() for r in agent.generations],
            ),
        )
        save_json(path / "organisms.json", [r.model_dump() for r in agent.organisms])
        save_json(path / "generations.json", [r.model_dump() for r in agent.generations])
        save_json(
            path / "leaderboard.json",
            [
                dict(
                    r.model_dump(exclude={"implementation", "calls", "revisions"}),
                    metrics=scorecards[r.policy_id],
                )
                for r in agent.elites
            ],
        )

    agent = MetaSearch(
        f"Evolve effective policy-search algorithms across {', '.join(config.environments)}. "
        f"Objective: {config.objective}. S is equal-weight symmetric percentage improvement "
        "over a measured fixed EliteTable optimizer with identical search budgets and seeds. "
        "Zero matches EliteTable; positive beats it. Per task: 200*(candidate-baseline)/"
        "(abs(candidate)+abs(baseline)), or zero if both are zero. "
        "T is mean output tokens; E is mean evaluated policies. Efficiency requires S >= 0.",
        editor,
        evaluate,
        context=CONTRACT
        + "\nTask specifications available to each sub-evolver:\n\n"
        + "\n\n".join(task_context(config, name) for name in config.environments),
        scorecards=scorecards,
        config=SearchConfig(
            population_size=config.population,
            generations=config.generations,
            elite_size=config.elites,
            max_repairs=config.max_repairs,
            generation_concurrency=config.model_workers,
            generation_timeout=config.generation_timeout,
        ),
        seed=config.search_seed,
        on_checkpoint=checkpoint,
    )
    previous = prompts.TEMPLATE_ROOT
    agent._call_slots = slots["models"]  # One global model-call limit, including the outer writer.
    prompts.TEMPLATE_ROOT = Path(__file__).parent.parent / "elitesearch" / "prompts"
    summary = dict(status="running", objective=config.objective)
    worker_report = asyncio.create_task(report_workers(slots))
    try:
        if (path / "checkpoint.json").exists():
            saved = read_json(path / "checkpoint.json")
            for card_path in (path / "development").glob("*/*/scorecard.json"):
                card = read_json(card_path)
                scorecards[card_path.parent.parent.name] = card
            agent.restore(
                [Organism.model_validate(r) for r in saved["organisms"]],
                [Generation.model_validate(r) for r in saved["generations"]],
            )
        baseline_for("development", config)  # Proposals start without waiting for baseline fitness.
        try:
            await agent.run()
        except BudgetExceeded:
            agent.reason = "budget_exhausted"
            agent._log_leaderboard()
            checkpoint(agent)
        if agent.best is None:
            raise RuntimeError("EliteTable produced no valid five-generation sub-evolver")
        winner = agent.best
        winner.to_file(path / "winner.py")
        save_json(
            path / "selection.json",
            dict(
                policy_id=winner.id,
                controller_hash=digest(winner._implementation),
                scorecard=scorecards[winner.id],
            ),
        )
        summary.update(policy_id=winner.id, stop=agent.reason, heldout={})
        progress(
            "Auditing frozen EliteTable winner",
            kind="search_started",
            optimizer="EliteTable meta",
            total_candidates=None,
        )
        for phase in ("validation", "test"):
            phase_config = config.model_copy(
                update={
                    "audit_cases": config.final_audit_cases
                    if phase == "test"
                    else config.audit_cases
                }
            )
            summary["heldout"][phase] = await benchmark(
                phase_config,
                winner._implementation,
                path / phase,
                getattr(config, phase),
                baseline_for(phase, phase_config),
                slots,
            )
        summary["status"] = "completed"
        progress(
            "EliteTable campaign completed",
            kind="search_finished",
            status="completed",
            reason=agent.reason,
        )
        return summary
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        worker_report.cancel()
        await asyncio.gather(worker_report, return_exceptions=True)
        for task in baseline_tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*baseline_tasks.values(), return_exceptions=True)
        prompts.TEMPLATE_ROOT = previous
        summary["editor"] = usage(editor)
        save_json(path / "summary.json", summary)


def campaign(config, path, baselines, loop):
    path.mkdir(exist_ok=True)
    with bind_run(path):
        progress(
            "Ocean sub-evolver search",
            kind="environment",
            name="Ocean " + ", ".join(config.environments),
        )
        return loop.run_until_complete(
            _campaign(config, path, baselines, baseline_root=path.parent / "baseline")
        )


def workload(config):
    tracks, tasks = (3 if config.objective == "all" else 1), len(config.environments)
    # A repaired sub-evolver is another complete evaluation, not a free trial.
    proposals = config.population * config.generations * (1 + config.max_repairs)
    development = proposals * len(config.development)
    baseline_trials = tasks * (len(config.development) + len(config.validation) + len(config.test))
    trials = (
        tracks * tasks * (development + len(config.validation) + len(config.test)) + baseline_trials
    )
    return dict(
        outer_candidates=config.population * config.generations,
        outer_generations=config.generations,
        inner_generations=5,
        max_policies_per_generation=config.generation_size,
        max_policies_per_trial=config.inner_generations * config.generation_size,
        max_evaluations_per_trial=config.evaluations,
        inner_repair_attempts=config.inner_max_repairs,
        search_games_per_evaluation=10,
        max_transitions_per_evaluation=10 * config.max_steps,
        max_controller_trials=trials,
        fixed_baseline_trials=baseline_trials,
        max_evaluated_proposals=trials * config.evaluations,
        max_search_games=trials * config.evaluations * 10,
        private_audit_games=tasks
        * (
            (len(config.development) + len(config.validation)) * config.audit_cases
            + len(config.test) * config.final_audit_cases
        )
        + tracks
        * tasks
        * (
            (development + len(config.validation)) * config.audit_cases
            + len(config.test) * config.final_audit_cases
        ),
        calibration_games=0,
        max_model_reservation_usd=trials * config.trial_spend_cap
        + tracks * config.editor_spend_cap,
        trial_workers=config.trial_workers,
        evaluation_workers=config.evaluation_workers,
    )
