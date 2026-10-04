"""EliteSearch's command options and environment-independent entry point."""

import asyncio
import json
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from slick import prompts
from sqlalchemy import inspect
from sqlmodel import select

from research.providers import CALL, BudgetExceeded
from rsikit import Policy, search

from .agent import Config, EliteSearch
from .records import Generation, Organism


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    population: int = Field(default=50, ge=1)
    generations: int = Field(default=20, ge=1)
    elites: int = Field(default=10, ge=1)
    max_repairs: int = Field(default=2, ge=0)
    new_fraction: float = Field(default=0.2, ge=0, le=1)
    remix_fraction: float = Field(default=0.4, ge=0, le=1)
    remix_parents: int = Field(default=3, ge=2)
    target_score: float | None = None

    @model_validator(mode="after")
    def valid_population(self):
        if self.elites > self.population:
            raise ValueError("elites must not exceed population")
        if self.new_fraction + self.remix_fraction > 1:
            raise ValueError("new_fraction + remix_fraction must not exceed 1")
        return self


def add_arguments(parser):
    for name in ("population", "generations", "elites", "max_repairs", "remix_parents"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("new_fraction", "remix_fraction", "target_score"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)


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


async def optimize(*, task, provider, evaluate, run, options, seed):
    # Runtime-only settings are supplied by the shared runner, outside public options.
    values = dict(options)
    generation = values.pop("generation")
    videos = values.pop("videos")
    validated = Options.model_validate(values)
    values = validated.model_dump()
    values["population_size"] = values.pop("population")
    values["elite_size"] = values.pop("elites")
    values.update(
        generation_concurrency=generation["concurrency"], generation_timeout=generation["timeout"]
    )
    previous_root = prompts.TEMPLATE_ROOT
    prompts.TEMPLATE_ROOT = Path(__file__).parent / "prompts"
    queue = asyncio.Queue()
    video_task = None
    reported = 0
    if videos["top"]:
        from .videos import generation_videos

        video_task = asyncio.create_task(
            generation_videos(queue, run.path, videos["top"], videos["workers"])
        )

    def checkpoint(agent):
        nonlocal reported
        run.save(*agent.records())
        if video_task is not None and video_task.done():
            video_task.result()
        completed = sum(g.status == "completed" for g in agent.generations)
        if completed > reported:
            reported = completed
            (run.path / "leaderboard.json").write_text(
                json.dumps(
                    [
                        row.model_dump(exclude={"implementation", "calls", "revisions"})
                        for row in agent.elites
                    ],
                    indent=2,
                )
                + "\n"
            )
            if video_task is not None:
                queue.put_nowait(agent.generations[-1].number)

    agent = Search(
        task,
        provider,
        context=task,
        config=Config(**values),
        seed=seed,
        on_checkpoint=checkpoint,
    )
    try:
        with run.database() as db:
            if inspect(db.get_bind()).has_table(Generation.__tablename__):
                agent.restore(list(db.exec(select(Organism))), list(db.exec(select(Generation))))
                reported = sum(g.status == "completed" for g in agent.generations)
        try:
            await search(agent, evaluate, on_checkpoint=checkpoint)
        except BudgetExceeded:
            agent.reason = "budget_exhausted"
            checkpoint(agent)
        if video_task is not None:
            queue.put_nowait(None)
            await video_task
        ranked = sorted(
            (row for row in agent.organisms if row.score is not None),
            key=lambda row: (-row.score, row.id),
        )
        policies = [
            Policy.from_text(row.implementation, name=row.name, description=row.description)
            for row in ranked
        ]
        return list({p.id: p for p in policies}.values())
    finally:
        if video_task is not None:
            video_task.cancel()
            await asyncio.gather(video_task, return_exceptions=True)
        prompts.TEMPLATE_ROOT = previous_root
