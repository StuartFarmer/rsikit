"""LineageSearch options and adapter for the common experiment runner."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from slick import prompts

from research.providers import BudgetExceeded
from rsikit import search

from .agent import Config, LineageSearch


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    families: int = Field(default=10, ge=1)
    initial_per_family: int = Field(default=10, ge=1)
    proposal_batch_size: int = Field(default=10, ge=1)
    frontier_per_family: int | None = Field(default=None, ge=1)
    cull_percent: float = Field(default=90.0, ge=0, lt=100)
    exploration: float = Field(default=0.2, ge=0, le=1)
    bonus_batches: int = Field(default=2, ge=0)
    patience: int = Field(default=3, ge=1)
    min_delta: float = Field(default=0.0, ge=0)
    uncertainty: float = Field(default=2.0, ge=0)
    max_attempts: int = Field(default=500, ge=0)
    discovery_attempts: int = Field(default=3, ge=1)
    max_repairs: int = Field(default=2, ge=0)
    decomposition_k: int = Field(default=3, ge=1)
    decomposition_max_votes: int = Field(default=40, ge=1)


def add_arguments(parser):
    for name, field in Options.model_fields.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int if name == "frontier_per_family" else field.annotation,
        )


async def optimize(*, task, provider, evaluate, run, options, seed):
    values = dict(options)
    generation = values.pop("generation")
    values.pop("videos")
    values = Options(**values).model_dump()
    values["batch_size"] = values.pop("proposal_batch_size")
    previous_root = prompts.TEMPLATE_ROOT
    prompts.TEMPLATE_ROOT = Path(__file__).parent / "prompts"

    def checkpoint(agent):
        run.save(*agent.records())
        for policy in agent._policies.values():
            run.save_policy(policy)

    agent = LineageSearch(
        task,
        provider,
        context=task,
        seed=seed,
        on_checkpoint=checkpoint,
        config=Config(
            **values,
            generation_concurrency=generation["concurrency"],
            generation_timeout=generation["timeout"],
        ),
    )
    try:
        try:
            await search(agent, evaluate, on_checkpoint=checkpoint)
        except BudgetExceeded:
            agent.study.reason = "budget_exhausted"
            checkpoint(agent)
        ranked = sorted(
            (r for r in agent.trials if r.score is not None), key=lambda r: (-r.score, r.id)
        )
        return list({r.policy_id: agent._policies[r.id] for r in ranked}.values())
    finally:
        prompts.TEMPLATE_ROOT = previous_root
