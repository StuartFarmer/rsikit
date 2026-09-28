"""Application search loops with automatic domain logging through Run."""

import asyncio
import logging
import math

from sqlalchemy import inspect
from sqlmodel import func, select

from research.alphaevolve import paper
from research.alphaevolve.history import Evaluation, Generation, history_records
from research.alphaevolve.paper.evaluation import assess
from research.rewards import mean_rewards
from rsikit.evaluation import PolicyError
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
    """Display completed policies immediately and keep the same messages in run.log."""
    if isinstance(generator, paper.AlphaEvolve):
        return await run_paper_search(
            generator,
            run,
            rollouts,
            generations=generations,
            batch_size=batch_size,
            generation_concurrency=generation_concurrency,
            seeds=seeds,
            screening_seeds=screening_seeds,
            screening_min_reward=screening_min_reward,
            initial_policy=initial_policy,
        )
    seeds = tuple(seeds)
    first_generation = 1
    with run.database() as db:
        if inspect(db.bind).has_table(Generation.__tablename__):
            first_generation = (db.exec(select(func.max(Generation.number))).one() or 0) + 1
    logger = logging.getLogger("rsikit")
    outcome = "failed"
    logger.info(
        "Starting search",
        extra={
            "progress": dict(
                kind="search_started",
                optimizer=type(generator).__module__,
                total_candidates=generations * batch_size + len(generator.attempts),
                total_generations=generations + generator._batch_number,
                columns=generator.leaderboard_columns,
                resumed=False,
            )
        },
    )
    try:
        logger.info("Run: %s", run.path)
        logger.info("Optimizer: %s", type(generator).__module__)
        for offset in range(generations):
            generation = first_generation + offset
            logger.info("Generation %s/%s", offset + 1, generations)
            history = dict(
                generation=generation,
                attempt_start=len(generator.attempts),
                event_start=len(generator.events),
                seeds=seeds,
            )
            complete, failures = (False, {})
            try:
                policies = await generator.generate(
                    n=batch_size, concurrency=generation_concurrency
                )
                scores = {}
                while policies:
                    run.save(*history_records(generator, **history, failures=failures))
                    try:
                        generator.evaluation_started(policies)
                        scores = await mean_rewards(rollouts, policies, seeds=seeds)
                        failures = {}
                        break
                    except PolicyError as exc:
                        if not exc.failures:
                            raise
                        failures = exc.failures
                        run.save(*history_records(generator, **history, failures=failures))
                        slots = asyncio.Semaphore(generation_concurrency)

                        async def repair(policy):
                            async with slots:
                                return await generator.repair(policy, failures[policy.id])

                        failed = {p.id: p for p in policies if p.id in failures}
                        repairs = {
                            id: asyncio.create_task(repair(policy)) for id, policy in failed.items()
                        }
                        try:
                            replacements = dict(
                                zip(repairs, await asyncio.gather(*repairs.values()))
                            )
                        finally:
                            for task in repairs.values():
                                task.cancel()
                            await asyncio.gather(*repairs.values(), return_exceptions=True)
                        policies = [
                            replacement
                            for policy in policies
                            if (replacement := replacements.get(policy.id, policy)) is not None
                        ]
                generator.update_scores(
                    scores,
                    seed_scores={
                        policy.id: {
                            seed: score
                            for seed, score in run.scores(policy).items()
                            if seed in seeds
                        }
                        for policy in policies
                    },
                )
                complete = True
            finally:
                run.save(
                    *history_records(generator, **history, complete=complete, failures=failures)
                )
            logger.info(
                "Finished generation",
                extra={
                    "progress": dict(
                        kind="batch_finished",
                        batch_id=str(generator._batch_number),
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


async def run_paper_search(
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
    """Overlap generation and evaluation; snapshot each evaluated batch.

    Attempts belong to their first-seen snapshot group, not synchronized generations.
    A group's complete flag waits for every assigned attempt to evaluate or discard;
    its island snapshot remains the population at the original batch boundary.
    """
    seeds, screening_seeds = (tuple(seeds), tuple(screening_seeds))
    _check_gym_evaluation(
        generator.config.features,
        seeds,
        screening_seeds,
        screening_min_reward,
        generator.config.objective,
    )
    if generator._attempt_offset:
        _log_history(run)
    logger = logging.getLogger("rsikit")
    generation = 1
    with run.database() as db:
        if inspect(db.bind).has_table(Generation.__tablename__):
            generation = (db.exec(select(func.max(Generation.number))).one() or 0) + 1
    next_attempt = len(generator.attempts)
    event_start = len(generator.events)
    unresolved = {}
    failures = {}
    failed_groups = set()
    failed_versions = {}

    async def evaluate(policies):
        nonlocal failures
        try:
            results = await assess(
                rollouts,
                policies,
                seeds=seeds,
                features=generator.config.features,
                screening_seeds=screening_seeds,
                screening_min_reward=screening_min_reward,
            )
            failures = {
                id: result.failure for id, result in results.items() if result.failure is not None
            }
            return results
        except PolicyError as exc:
            failures = exc.failures
            raise

    def persist(event, policies):
        nonlocal next_attempt, event_start, generation
        for record in generator.attempts[next_attempt:]:
            unresolved[record["id"]] = (generation, record)
        next_attempt = len(generator.attempts)
        batches = {generation: []}
        for number, record in unresolved.values():
            batches.setdefault(number, []).append(record)
            policy = record.get("policy")
            version = (record["id"], record.get("revision", 0))
            if event == "evaluation_failed" and policy is not None and (policy.id in failures):
                failed_versions[version] = failures[policy.id]
            if record["status"] in ("cancelled", "rejected", "error"):
                failed_groups.add(number)
        for number, attempts in batches.items():
            complete = (
                (bool(attempts) or event == "evaluated")
                and number not in failed_groups
                and all((row["status"] in ("evaluated", "discarded") for row in attempts))
            )
            rows = history_records(
                generator,
                generation=number,
                attempt_start=0,
                attempts=attempts,
                event_start=event_start,
                seeds=seeds,
                complete=complete,
                failures={
                    row["policy"].id: failed_versions[version]
                    for row in attempts
                    if (version := (row["id"], row.get("revision", 0))) in failed_versions
                },
            )
            if number != generation:
                rows.pop()
                if complete:
                    with run.database() as db:
                        snapshot = db.get(Generation, number)
                    snapshot.complete = True
                    rows.append(snapshot)
            run.save(*rows)
        unresolved_keys = [
            key
            for key, (_, row) in unresolved.items()
            if row["status"] not in ("generating", "generated", "repaired", "evaluating")
        ]
        for key in unresolved_keys:
            del unresolved[key]
        if event == "evaluated":
            if generator.best is not None:
                logger.info("Best so far: %s", generator.best.name)
            generation += 1
            event_start = len(generator.events)

    try:
        logger.info("Run: %s", run.path)
        logger.info("Optimizer: %s", type(generator).__module__)
        if initial_policy is not None:
            policy = Policy.from_file(initial_policy)
            validate_policy(policy)
            result = (await evaluate([policy]))[policy.id]
            if not result.accepted:
                raise ValueError("Initial policy failed screening; it was not registered")
            generator.register_initial(policy, result)
        await paper.search(
            generator,
            evaluate,
            proposals=generations * batch_size,
            generation_concurrency=generation_concurrency,
            evaluation_batch_size=batch_size,
            on_event=persist,
        )
    except Exception:
        logger.exception("Run failed; saved results and details are in %s", run.path)
        raise
    finally:
        if unresolved or next_attempt < len(generator.attempts):
            persist("finished", [])
        generator.checkpoint()
