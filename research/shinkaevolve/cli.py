"""ShinkaEvolve options and adapter for the common experiment runner."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from slick import prompts

from research.providers import BudgetExceeded
from rsikit import search

from .agent import Config, ShinkaEvolve


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    proposal_batch_size: int = Field(default=25, ge=1)
    generations: int = Field(default=10, ge=0)
    islands: int = Field(default=2, ge=1)
    archive_size: int = Field(default=40, ge=1)
    elite_ratio: float = Field(default=0.3, ge=0, le=1)
    top_k: int = Field(default=2, ge=1)
    inspirations: int = Field(default=4, ge=0)
    parent_selection: Literal["weighted", "uniform", "best", "power"] = "weighted"
    selection_pressure: float = Field(default=10.0, ge=0)
    power_alpha: float = Field(default=1.0, ge=0)
    exploration: float = Field(default=1.0, ge=0)
    max_proposals: int = Field(default=3, ge=1)
    max_repairs: int = Field(default=2, ge=0)
    novelty_threshold: float = Field(default=0.95, ge=-1, le=1)
    meta_interval: int = Field(default=10, ge=0)
    max_recommendations: int = Field(default=5, ge=0)
    migration_interval: int = Field(default=10, ge=0)
    migration_rate: float = Field(default=0.1, ge=0, le=1)


def add_arguments(parser):
    for name, field in Options.model_fields.items():
        if name == "parent_selection":
            parser.add_argument(
                "--parent-selection", choices=("weighted", "uniform", "best", "power")
            )
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=field.annotation)


async def optimize(*, task, provider, evaluate, run, options, seed):
    values = dict(options)
    generation = values.pop("generation")
    values.pop("videos")
    values = Options(**values).model_dump()
    values["batch_size"] = values.pop("proposal_batch_size")
    previous_root = prompts.TEMPLATE_ROOT
    prompts.TEMPLATE_ROOT = Path(__file__).parent / "prompts"
    agent = ShinkaEvolve(
        task,
        provider,
        context=task,
        seed=seed,
        config=Config(
            **values,
            generation_concurrency=generation["concurrency"],
            generation_timeout=generation["timeout"],
        ),
    )

    def checkpoint(agent):
        if agent.generations:
            agent.records(
                seeds=sorted(agent._seed_panel or ()), complete=agent.generations[-1].complete
            )
            run.save(*agent.evaluations, *agent.generations)
        for policy in agent._policies.values():
            run.save_policy(policy)

    try:
        try:
            await search(agent, evaluate, on_checkpoint=checkpoint)
        except BudgetExceeded:
            checkpoint(agent)
        ranked = sorted(
            (r for r in agent.evaluations if r.score is not None),
            key=lambda r: (-r.score, r.attempt),
        )
        return list({r.policy_id: agent._policies[r.policy_id] for r in ranked}.values())
    finally:
        prompts.TEMPLATE_ROOT = previous_root
