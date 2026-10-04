"""Application search loops with automatic domain logging through Run."""

import asyncio
import logging
import math
from dataclasses import replace

from sqlalchemy import inspect
from sqlmodel import func, select

from research.alphaevolve import paper
from research.alphaevolve.history import Evaluation, Generation, history_records
from research.alphaevolve.paper.evaluation import assess
from research.rewards import measure_rewards
from rsikit import search
from rsikit.policy import Policy, validate_policy


def _log_history(run):
    """Re-emit saved optimizer evidence before continuing its search."""
    with run.database() as db:
        if not inspect(db.bind).has_table(Evaluation.__tablename__):
            return
        latest = {}
        for row in db.exec(select(Evaluation).order_by(Evaluation.revision)):
            latest[row.attempt] = row
    logger = logging.getLogger(__name__)
    policies = {policy.id: policy for policy in run.policies()}
    for number in sorted({row.generation for row in latest.values()}):
        rows = [row for row in latest.values() if row.generation == number]
        batch = f"history-{number}"
        logger.info(
            "Restoring generation %s",
            number,
            extra={
                "progress": dict(
                    kind="batch_started",
                    batch_id=batch,
                    label=f"Generation {number}",
                    total_candidates=len(rows),
                )
            },
        )
        for row in rows:
            policy = policies.get(row.policy_id)
            status = {"repaired": "generated", "rejected": "failed", "error": "failed"}.get(
                row.status, row.status
            )
            logger.info(
                "Restored attempt %s: %s",
                row.attempt,
                status,
                extra={
                    "progress": dict(
                        kind="candidate",
                        batch_id=batch,
                        attempt_id=str(row.attempt),
                        revision=row.revision,
                        status=status,
                        proposal_done=row.policy_id is not None,
                        policy_id=row.policy_id or "—",
                        name=policy.name if policy else "",
                        description=policy.description if policy else "",
                        score=row.score,
                        restored=True,
                    )
                },
            )
        logger.info(
            "Restored generation %s",
            number,
            extra={
                "progress": dict(
                    kind="batch_finished",
                    batch_id=batch,
                    status="completed"
                    if all(row.status in ("evaluated", "discarded") for row in rows)
                    else "stopped",
                    restored=True,
                )
            },
        )


def _check_gym_evaluation(
    features, seeds, screening_seeds, screening_min_reward, objective="reward"
):
    if not seeds:
        raise ValueError("Evaluation requires at least one seed")
    if bool(screening_seeds) != (screening_min_reward is not None):
        raise ValueError("screening_seeds and screening_min_reward must be supplied together")
    if screening_min_reward is not None and (not math.isfinite(screening_min_reward)):
        raise ValueError("screening_min_reward must be finite")
    unknown = set(features) - {"mean_reward", "reward_std"}
    if unknown:
        raise ValueError(f"Unsupported Gym descriptors: {', '.join(sorted(unknown))}")
    if objective not in ("reward", "worst_reward", "stability"):
        raise ValueError(f"Unsupported Gym objective: {objective}")


async def run_search(
    generator,
    run,
    rollouts,
    *,
    generations,
    batch_size,
    generation_concurrency=4,
    seeds=(0,),
    screening_seeds=(),
    screening_min_reward=None,
    initial_policy=None,
):
    """Configure the shared runner and persist algorithm-owned history."""
    seeds = tuple(seeds)
    is_paper = isinstance(generator, paper.AlphaEvolve)
    if is_paper:
        _check_gym_evaluation(
            generator.config.features,
            seeds,
            screening_seeds,
            screening_min_reward,
            generator.config.objective,
        )
        if generator._attempt_offset or generator.attempts:
            _log_history(run)
    generator.config = replace(
        generator.config,
        proposals=generator._attempt_offset + len(generator.attempts) + generations * batch_size,
        batch_size=batch_size,
        generation_concurrency=generation_concurrency,
    )
    with run.database() as db:
        last_generation = (
            db.exec(select(func.max(Generation.number))).one() or 0
            if inspect(db.bind).has_table(Generation.__tablename__)
            else 0
        )
    offset = max(0, last_generation - generator._batch_number)
    event_starts = {}
    logger = logging.getLogger("rsikit")
    logger.info("Run: %s", run.path)
    logger.info("Optimizer: %s", type(generator).__module__)
    logger.info(
        "Starting search",
        extra={
            "progress": dict(
                kind="search_started",
                optimizer=type(generator).__module__,
                total_candidates=generator.config.proposals,
                columns=generator.leaderboard_columns,
            )
        },
    )

    async def evaluate(policies):
        if is_paper:
            return await assess(
                rollouts,
                policies,
                seeds=seeds,
                features=generator.config.features,
                screening_seeds=screening_seeds,
                screening_min_reward=screening_min_reward,
            )
        return await measure_rewards(rollouts, policies, seeds=seeds)

    def checkpoint(agent):
        batch = agent._batch_number
        rows = [row for row in agent.attempts if row.get("batch") == batch]
        if rows:
            start = event_starts.setdefault(batch, len(agent.events))
            complete = all(row["status"] in ("evaluated", "discarded") for row in rows)
            for row in rows:
                if row.get("policy") is not None:
                    run.save_policy(row["policy"])
            run.save(
                *history_records(
                    agent,
                    generation=offset + batch,
                    attempt_start=0,
                    attempts=rows,
                    event_start=start,
                    seeds=seeds,
                    complete=complete,
                    failures={
                        row["policy"].id: row["error"]
                        for row in rows
                        if row["status"] == "execution_failed"
                    },
                )
            )
            if complete:
                logger.info(
                    "Finished generation",
                    extra={
                        "progress": dict(
                            kind="batch_finished", batch_id=str(batch), status="completed"
                        )
                    },
                )
        if is_paper:
            agent.checkpoint()

    outcome = "failed"
    try:
        if initial_policy is not None:
            policy = Policy.from_file(initial_policy)
            validate_policy(policy)
            measurement = (await evaluate([policy]))[policy.id]
            if not measurement.accepted:
                raise ValueError("Initial policy failed screening; it was not registered")
            generator.register_initial(policy, generator._evaluation_result(measurement))
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


# Historical spelling; all variants now share the same application loop.
run_paper_search = run_search
