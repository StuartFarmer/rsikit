"""Configure and persist ShinkaEvolve around the shared round runner."""

import asyncio
import logging
from dataclasses import replace

from research.rewards import measure_rewards
from rsikit import search


async def run_search(
    generator, run, rollouts, *, generations, batch_size, generation_concurrency=4, seeds=(0,)
):
    seeds = tuple(seeds)
    generator.config = replace(
        generator.config,
        generations=len(generator.generations) + generations,
        batch_size=batch_size,
        generation_concurrency=generation_concurrency,
    )
    logger = logging.getLogger("research.shinkaevolve")
    logger.info("Run: %s", run.path)
    logger.info("Optimizer: %s", type(generator).__module__)
    logger.info(
        "Starting search",
        extra={
            "progress": dict(
                kind="search_started",
                optimizer="ShinkaEvolve",
                total_candidates=generations * batch_size + generator._attempt,
                columns=generator.leaderboard_columns,
            )
        },
    )

    async def evaluate(policies):
        return await measure_rewards(rollouts, policies, seeds=seeds)

    def checkpoint(agent):
        if agent.generations:
            agent.records(seeds=seeds, complete=agent.generations[-1].complete)
            for generation in agent.generations:
                generation.seeds = list(seeds)
            # ponytail: rewrite small run histories; persist dirty records if long searches need it.
            run.save(*agent.evaluations, *agent.generations)

    outcome = "failed"
    try:
        await search(generator, evaluate, on_checkpoint=checkpoint)
        outcome = "completed"
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    finally:
        logger.info(
            "Search %s",
            outcome,
            extra={"progress": dict(kind="search_finished", status=outcome, reason=outcome)},
        )
