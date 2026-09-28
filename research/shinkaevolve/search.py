"""Application search loops with automatic domain logging through Run."""

import asyncio
import logging

from research.rewards import mean_rewards
from rsikit.evaluation import PolicyError


async def run_search(
    generator, run, rollouts, *, generations, batch_size, generation_concurrency=4, seeds=(0,)
):
    seeds = tuple(seeds)
    logger = logging.getLogger("research.shinkaevolve")
    outcome = "failed"
    logger.info(
        "Starting search",
        extra={
            "progress": dict(
                kind="search_started",
                optimizer=type(generator).__module__,
                total_candidates=generations * batch_size + generator._attempt,
                total_generations=generations + len(generator.generations),
                columns=generator.leaderboard_columns,
                resumed=False,
            )
        },
    )
    try:
        logger.info("Run: %s", run.path)
        logger.info("Optimizer: %s", type(generator).__module__)
        for generation in range(generations):
            logger.info("Generation %s/%s", generation + 1, generations)
            complete = False
            try:
                policies = await generator.generate(
                    n=batch_size, concurrency=generation_concurrency
                )
                scores = {}
                while policies:
                    run.save(*generator.records(seeds=seeds))
                    try:
                        generator.evaluation_started(policies)
                        scores = await mean_rewards(rollouts, policies, seeds=seeds)
                        break
                    except PolicyError as exc:
                        if not exc.failures:
                            raise
                        generator.evaluation_failed(exc.failures)
                        run.save(*generator.records(seeds=seeds))
                        replacements = {}
                        for policy in policies:
                            if policy.id in exc.failures and policy.id not in replacements:
                                replacements[policy.id] = await generator.repair(
                                    policy, exc.failures[policy.id]
                                )
                        policies = [
                            replacement
                            for policy in policies
                            if (replacement := replacements.get(policy.id, policy)) is not None
                        ]
                generator.update(scores)
                complete = True
            finally:
                run.save(*generator.records(seeds=seeds, complete=complete))
            logger.info(
                "Finished generation",
                extra={
                    "progress": dict(
                        kind="batch_finished",
                        batch_id=str(len(generator.generations)),
                        status="completed",
                    )
                },
            )
            if not policies:
                logger.warning("No surviving policies in this generation; continuing")
            if generator.best is not None:
                logger.info("Best so far: %s", generator.best.name)
        outcome = "completed"
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except Exception:
        logger.exception("Run failed; saved results and details are in %s", run.path)
        raise
    finally:
        logger.info(
            "Search %s",
            outcome,
            extra={"progress": dict(kind="search_finished", status=outcome, reason=outcome)},
        )
