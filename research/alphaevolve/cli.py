"""AlphaEvolve options and adapter for the common experiment runner."""

import logging
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from slick import prompts

from research.providers import BudgetExceeded
from rsikit import search

from . import improved, original, paper
from .history import history_records


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    variant: Literal["paper", "original", "improved"] = "paper"
    proposal_batch_size: int = Field(default=10, ge=1)
    proposals: int = Field(default=250, ge=0)
    islands: int = Field(default=4, ge=1)
    inspirations: int = Field(default=3, ge=0)
    exploration: float | None = Field(default=None, ge=0, le=1)
    reset_interval: int | None = Field(default=None, ge=0)
    meta_interval: int | None = Field(default=None, ge=0)
    mode: Literal["diff", "rewrite"] = "diff"
    max_repairs: int = Field(default=2, ge=0)
    objective: Literal["reward", "worst_reward", "stability"] | None = None
    elite_fraction: float | None = Field(default=None, gt=0, le=1)
    migration_interval: int | None = Field(default=None, ge=0)
    migration_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_variant(self):
        self.configuration()
        return self

    def configuration(self, **runtime):
        values = self.model_dump(exclude_none=True)
        variant = values.pop("variant")
        values["batch_size"] = values.pop("proposal_batch_size")
        if (
            variant != "paper"
            and {"objective", "elite_fraction", "migration_interval", "migration_count"}
            & values.keys()
        ):
            raise ValueError("Archive options require variant=paper")
        return {"paper": paper, "original": original, "improved": improved}[variant].Config(
            **values, **runtime
        )


def add_arguments(parser):
    parser.add_argument("--variant", choices=("paper", "original", "improved"))
    parser.add_argument("--mode", choices=("diff", "rewrite"))
    parser.add_argument("--objective", choices=("reward", "worst_reward", "stability"))
    for name in (
        "proposal_batch_size",
        "proposals",
        "islands",
        "inspirations",
        "reset_interval",
        "meta_interval",
        "max_repairs",
        "migration_interval",
        "migration_count",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("exploration", "elite_fraction"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)


async def optimize(*, task, provider, evaluate, run, options, seed):
    values = dict(options)
    generation = values.pop("generation")
    values.pop("videos")
    options = Options(**values)
    variant = {"paper": paper, "original": original, "improved": improved}[options.variant]
    config = options.configuration(
        generation_concurrency=generation["concurrency"], generation_timeout=generation["timeout"]
    )
    previous_root = prompts.TEMPLATE_ROOT
    prompts.TEMPLATE_ROOT = Path(__file__).parent
    agent = None
    saved = set()

    def checkpoint(agent):
        for batch in sorted({r["batch"] for r in agent.attempts} - saved):
            rows = [r for r in agent.attempts if r["batch"] == batch]
            complete = all(r["status"] in ("evaluated", "discarded") for r in rows)
            for row in rows:
                if row.get("policy") is not None:
                    run.save_policy(row["policy"])
            run.save(
                *history_records(
                    agent,
                    generation=batch,
                    attempt_start=0,
                    event_start=0,
                    attempts=rows,
                    seeds=sorted(agent._seed_panel or ()),
                    complete=complete,
                )
            )
            if complete:
                saved.add(batch)
        if options.variant == "paper":
            agent.checkpoint()

    try:
        agent = variant.AlphaEvolve(
            task,
            provider,
            context=task,
            config=config,
            seed=seed,
            **(
                {"database_path": run.path / "population.sqlite"}
                if options.variant == "paper"
                else {}
            ),
        )
        try:
            await search(agent, evaluate, on_checkpoint=checkpoint)
        except BudgetExceeded:
            checkpoint(agent)
        ranked = sorted(
            (r for r in agent.attempts if r.get("score") is not None),
            key=lambda r: (-r["score"], r["id"]),
        )
        return list({r["policy"].id: r["policy"] for r in ranked}.values())
    finally:
        primary_error = sys.exc_info()[0] is not None
        try:
            if options.variant == "paper" and agent is not None:
                try:
                    agent.close()
                except BaseException:
                    if not primary_error:
                        raise
                    logging.getLogger(__name__).exception(
                        "Cleanup failed while handling search error"
                    )
        finally:
            prompts.TEMPLATE_ROOT = previous_root
