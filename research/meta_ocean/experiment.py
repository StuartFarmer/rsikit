"""Trusted budgets, split definitions and trial accounting for the Ocean suite."""

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import Field, model_validator

from research.experiment import Options, save_json
from research.providers import BudgetExceeded, BudgetProvider, UsageOpenRouter
from rsikit import Policy
from rsikit.policy import MAX_SOURCE, validate_policy


def progress(message, **payload):
    logging.getLogger(__name__).info(message, extra={"progress": payload} if payload else {})


class Config(Options):
    model: str = "gpt-oss-120b:nitro"
    image: str = "rsikit-meta-worker:local"
    objective: Literal["performance", "tokens", "evaluations", "all"] = "all"
    environments: list[Literal["g2048", "breakout", "maze"]] = Field(
        default_factory=lambda: ["g2048", "breakout", "maze"], min_length=1
    )
    development: list[int] = Field(default_factory=lambda: [0, 1])
    validation: list[int] = Field(default_factory=lambda: [2, 3, 4])
    test: list[int] = Field(default_factory=lambda: [5, 6, 7, 8, 9])
    evaluations: int = Field(default=50, ge=1, le=10000)
    audit_cases: int = Field(default=64, ge=1, le=512)
    final_audit_cases: int = Field(default=512, ge=1, le=512)
    calibration_cases: int = Field(default=64, ge=1, le=512)
    batch_size: int = Field(default=32, ge=1, le=32)
    max_steps: int = Field(default=2000, ge=1, le=2000)
    benchmark_limit: int = Field(default=9, ge=1)
    revisions: int = Field(default=2, ge=1)
    search_seed: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=250000, ge=1)
    max_input_tokens: int = Field(default=65536, ge=1024)
    max_output_tokens: int = Field(default=4096, ge=1)
    editor_output_tokens: int = Field(default=16384, ge=1)
    input_price: float = Field(default=1, gt=0)
    output_price: float = Field(default=5, gt=0)
    trial_spend_cap: float = Field(default=5, gt=0)
    editor_spend_cap: float = Field(default=5, gt=0)
    trial_timeout: float = Field(default=7200, gt=0)
    panel_timeout: float = Field(default=1800, gt=0)
    generation_timeout: float = Field(default=120, gt=0)

    @model_validator(mode="after")
    def valid_panels(self):
        if len(self.environments) != len(set(self.environments)):
            raise ValueError("Environments must be unique")
        panels = [self.development, self.validation, self.test]
        seeds = [seed for panel in panels for seed in panel]
        if any(not panel for panel in panels) or len(seeds) != len(set(seeds)):
            raise ValueError(
                "Development, validation and test replicates must be nonempty/disjoint"
            )
        if any(type(s) is not int or not 0 <= s < 100000 for s in seeds):
            raise ValueError("Replicate IDs must be integers in [0, 100000)")
        if self.max_output_tokens > self.output_tokens:
            raise ValueError("Per-call output cap exceeds trial output budget")
        return self


def append_json(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def source_text(value):
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > MAX_SOURCE:
        raise ValueError("Source must be nonempty text of at most 64 KiB")
    # Unwrap one unambiguous code block; never guess between alternative programs.
    blocks = re.findall(r"^```(?:python|py)?[ \t]*\n(.*?)^```[ \t]*$", value, re.M | re.S)
    if len(blocks) == 1:
        return blocks[0].rstrip() + "\n"
    return value


def usage(provider):
    events = provider.events

    def total(key):
        values = [(event.get("usage") or {}).get(key) for event in events]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            return None
        return sum(values)

    return dict(
        calls=provider.calls,
        input_tokens=total("prompt_tokens"),
        output_tokens=total("completion_tokens"),
        actual_cost=sum(e["actual_cost"] for e in events)
        if all(e.get("actual_cost") is not None for e in events)
        else None,
        reserved_tokens=provider.reserved_tokens,
        reserved_cost=provider.reserved_cost,
    )


def provider(config, path, *, editor=False):
    cap = config.editor_output_tokens if editor else config.max_output_tokens
    # Conservative reservations are never refunded; unknown usage cannot buy more calls.
    calls = (
        config.revisions
        if editor
        else min(
            config.evaluations * 3,
            config.output_tokens // cap,
            10 * config.output_tokens // config.max_input_tokens,
        )
    )
    if calls < 1:
        raise ValueError("Token ceilings cannot admit even one model call")

    def log(event):
        append_json(path, event)
        label = "GEPA editor" if editor else "Controller model"
        progress(
            f"{label} call {event['call']}/{calls}: {event['status']}; "
            f"usage={event['usage']}, cost={event['actual_cost']}"
        )

    return BudgetProvider(
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


def scorecard(rows, *, baseline, reference, objective):
    if not math.isfinite(baseline) or not math.isfinite(reference) or reference <= baseline:
        raise ValueError("Calibration requires a finite reference score above the starter")
    score = 100 * (fmean(row["score"] for row in rows) - baseline) / (reference - baseline)
    tokens = (
        None
        if any(r["output_tokens"] is None for r in rows)
        else fmean(r["output_tokens"] for r in rows)
    )
    evaluations = fmean(row["evaluations"] for row in rows)
    token_ratio = 1_000_000 * score / tokens if tokens else None
    evaluation_ratio = 100 * score / evaluations if evaluations else None
    qualified = score >= 100
    ratio = token_ratio if objective == "tokens" else evaluation_ratio
    if objective == "performance":
        selection = score
    elif ratio is None:
        selection = -1e30
    elif not qualified:
        selection = min(score - 100, 0)
    else:
        # Every qualified ratio outranks every unqualified entry; order within each is preserved.
        selection = 1 + ratio / (1 + ratio)
    return dict(
        S=score,
        T=tokens,
        E=evaluations,
        token_efficiency=token_ratio,
        evaluation_efficiency=evaluation_ratio,
        qualified=qualified,
        selection_score=selection,
    )


def suite_scorecard(rows, anchors, objective):
    """Normalize each task first; weight tasks equally, including resource means."""
    per_environment = {
        name: scorecard(
            [row for row in rows if row["environment"] == name],
            **anchor,
            objective=objective,
        )
        for name, anchor in anchors.items()
    }
    result = scorecard(
        [
            dict(score=r["S"], output_tokens=r["T"], evaluations=r["E"])
            for r in per_environment.values()
        ],
        baseline=0,
        reference=100,
        objective=objective,
    )
    result["environments"] = per_environment
    return result


class Trial:
    """The controller can request only metered generation, evaluation and commitment."""

    def __init__(self, config, path, starter, evaluate, model, replicate):
        self.config, self.path = config, path
        self.evaluate, self.model = evaluate, model
        self.seeds = list(range(replicate * 10000, replicate * 10000 + 10))
        self.incumbent = starter
        self.accepted = {}
        self.evaluations = self.transitions = 0
        self.failures = 0

    async def handle(self, request):
        op = request.get("op") if isinstance(request, dict) else None
        value = request.get("value") if isinstance(request, dict) else None
        try:
            if op == "evaluate":
                if self.evaluations >= self.config.evaluations:
                    raise BudgetExceeded("evaluation cap")
                self.evaluations += 1  # Charge before parsing, including duplicates and failures.
                append_json(
                    self.path / "oracle.jsonl",
                    dict(event="evaluation_admitted", number=self.evaluations),
                )
                number = self.evaluations
                progress(f"{self.path}: evaluation {number}/{self.config.evaluations}")
                try:
                    source = source_text(value)
                    policy = Policy.from_text(source)
                    validate_policy(policy)
                    (self.path / f"proposal-{number}.py").write_text(source)
                    result = await self.evaluate(source, self.seeds)
                    self.transitions += result["steps"]
                    identifier = digest(source)
                    self.accepted[identifier] = source
                    response = dict(id=identifier, score=result["score"], scores=result["scores"])
                except BaseException as exc:
                    self.failures += 1
                    progress(f"Evaluation {number} failed: {type(exc).__name__}: {exc}")
                    raise
            elif op == "generate":
                if not isinstance(value, str):
                    raise ValueError("Model prompt must be text")
                text, _ = await self.model.acall(value)
                response = dict(text=text)
            elif op == "commit":
                if not isinstance(value, str) or value not in self.accepted:
                    raise ValueError("Commit must name a successfully evaluated policy ID")
                self.incumbent = self.accepted[value]
                (self.path / "incumbent.py").write_text(self.incumbent)
                response = dict(committed=value)
            else:
                raise ValueError("Unknown controller operation")
        except Exception as exc:
            response = dict(error=f"{type(exc).__name__}: {exc}")
        append_json(self.path / "oracle.jsonl", dict(request=request, response=response))
        return response

    def finish(self, error=None):
        (self.path / "incumbent.py").write_text(self.incumbent)
        result = dict(
            evaluations=self.evaluations,
            successful_evaluations=self.evaluations - self.failures,
            failures=self.failures,
            completed_transitions=self.transitions,
            incomplete_transitions_unknown=bool(self.failures),
            error=error,
            **usage(self.model),
        )
        save_json(self.path / "search.json", result)
        progress(
            f"Controller finished: {self.evaluations} evaluations, {self.failures} failures, "
            f"{result['output_tokens']} output tokens; error={error}"
        )
        return result
