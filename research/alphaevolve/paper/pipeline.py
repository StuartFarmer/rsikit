"""Bounded, overlapping program generation and evaluation (paper §2.6)."""

import asyncio
import logging
from collections.abc import Mapping

from rsikit.evaluation import PolicyError

from .evaluation import EvaluationResult


async def search(
    generator,
    evaluate_batch,
    *,
    proposals: int,
    generation_concurrency: int = 4,
    evaluation_batch_size: int = 10,
    on_event=None,
) -> None:
    """Attempt a fixed budget, seed selection, then overlap bounded workers.

    evaluate_batch(policies) returns policy-ID -> EvaluationResult. Repairs use
    the generator's own repair budget and do not consume proposal attempts.
    Runtime repairs and proposal workers share generation_concurrency slots.
    on_event(event, policies) is synchronous so persistence also runs when the
    search is cancelled. Events are generated, repair, evaluated, discarded,
    evaluation_failed, and failed; failed is emitted after worker cleanup.
    """
    for name, value, minimum in (
        ("proposals", proposals, 0),
        ("generation_concurrency", generation_concurrency, 1),
        ("evaluation_batch_size", evaluation_batch_size, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    from .agent import AlphaEvolve

    reporting = isinstance(generator, AlphaEvolve)
    logger = logging.getLogger(__name__)
    offset = generator._attempt_offset + len(generator.attempts) if reporting else 0
    if reporting:
        generator._streaming_progress = (offset, evaluation_batch_size, proposals)
    logger.info(
        "Starting AlphaEvolve paper search",
        extra={
            "progress": dict(
                kind="search_started",
                optimizer="AlphaEvolve (paper)",
                total_candidates=proposals + offset,
                completed_candidates_before=generator._completed_offset if reporting else 0,
                columns=generator.leaderboard_columns if reporting else {},
                resumed=offset > 0,
            )
        },
    )
    if reporting:
        generator._log_leaderboard()
    closed_groups = set()
    outcome = "completed"
    pending = {}
    tasks = []
    slots = asyncio.Semaphore(generation_concurrency)

    def emit(event, policies):
        if on_event is not None:
            on_event(event, list(policies))
        if reporting:
            groups = {}
            for row in generator.attempts:
                if row["id"] > offset:
                    groups.setdefault(row["batch"], []).append(row)
            for group, rows in groups.items():
                index = (rows[0]["id"] - offset - 1) // evaluation_batch_size
                expected = min(evaluation_batch_size, proposals - index * evaluation_batch_size)
                if (
                    group not in closed_groups
                    and len(rows) == expected
                    and all(row["status"] in ("evaluated", "discarded") for row in rows)
                ):
                    logger.info(
                        "Finished batch %s",
                        group,
                        extra={
                            "progress": dict(
                                kind="batch_finished", batch_id=str(group), status="completed"
                            )
                        },
                    )
                    closed_groups.add(group)

    def ready(policies, event="generated"):
        pending.update((policy.id, policy) for policy in policies)
        emit(event if policies else "discarded", policies)

    async def repair(policy, diagnostic):
        async with slots:
            replacement = await generator.repair(policy, diagnostic)
        pending.pop(policy.id, None)
        if replacement is None:
            # repair() records exhaustion and removes the pending proposal.
            emit("discarded", [policy])
        else:
            ready([replacement], "repair")
        return replacement

    async def evaluate(policies):
        # One result updates every pending attempt for an identical program.
        # Copies already consumed through a concurrent batch can remain queued.
        policies = list({policy.id: policy for policy in policies if policy.id in pending}.values())
        measured = {}
        while policies:
            batch = [policy for policy in policies if policy.id not in measured]
            try:
                if reporting:
                    generator.evaluation_started(batch)
                results = await evaluate_batch(batch) if batch else {}
                if not isinstance(results, Mapping) or set(results) != {p.id for p in batch}:
                    raise ValueError("Evaluator must return exactly the evaluated policy IDs")
                if any(not isinstance(result, EvaluationResult) for result in results.values()):
                    raise ValueError("Evaluator must return EvaluationResult values")
                measured.update(
                    (id, result) for id, result in results.items() if result.failure is None
                )
                failures = {
                    id: result.failure
                    for id, result in results.items()
                    if result.failure is not None
                }
                if failures:
                    error = PolicyError("Candidate evaluation failed")
                    error.failures = failures
                    raise error
            except PolicyError as exc:
                if not exc.failures or not set(exc.failures) <= {p.id for p in batch}:
                    raise
                emit("evaluation_failed", [p for p in policies if p.id in exc.failures])
                repairs = {
                    policy.id: asyncio.create_task(repair(policy, exc.failures[policy.id]))
                    for policy in policies
                    if policy.id in exc.failures
                }
                try:
                    await asyncio.gather(*repairs.values())
                finally:
                    for task in repairs.values():
                        task.cancel()
                    await asyncio.gather(*repairs.values(), return_exceptions=True)
                replacements = {id: task.result() for id, task in repairs.items()}
                survivors = [
                    replacement
                    for policy in policies
                    if (replacement := replacements.get(policy.id, policy)) is not None
                ]
                policies = list({policy.id: policy for policy in survivors}.values())
                continue
            results = {policy.id: measured[policy.id] for policy in policies}
            accepted = {id: result for id, result in results.items() if result.accepted}
            if accepted:
                generator.update_results(accepted)
            for policy in policies:
                result = results[policy.id]
                if not result.accepted:
                    generator.discard(policy, result.feedback or "Evaluation threshold rejected")
                    emit("discarded", [policy])
                pending.pop(policy.id, None)
            emit("evaluated", policies)
            return

    try:
        if not proposals:
            return
        # Founders must be measured before later proposals can sample the archive.
        initial = min(proposals, evaluation_batch_size)
        policies = await generator.generate(initial, concurrency=generation_concurrency)
        ready(policies)
        await evaluate(policies)
        remaining = iter(range(proposals - initial))
        queue = asyncio.Queue(maxsize=evaluation_batch_size)
        workers = min(generation_concurrency, proposals - initial)

        async def produce():
            for _ in remaining:
                async with slots:
                    policies = await generator.generate(1, concurrency=1)
                if len(policies) > 1:
                    raise ValueError("generate(1) returned more than one policy")
                ready(policies)
                for policy in policies:
                    await queue.put(policy)
            await queue.put(None)

        async def consume():
            finished = 0
            while finished < workers:
                policy = await queue.get()
                if policy is None:
                    finished += 1
                    continue
                batch = [policy]
                while len(batch) < evaluation_batch_size:
                    try:
                        policy = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if policy is None:
                        finished += 1
                    else:
                        batch.append(policy)
                await evaluate(batch)

        tasks = [asyncio.create_task(produce()) for _ in range(workers)]
        tasks.append(asyncio.create_task(consume()))
        await asyncio.gather(*tasks)
    except BaseException as exc:
        outcome = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        emit("failed", pending.values())
        raise
    finally:
        groups = (
            dict.fromkeys(row.get("batch") for row in generator.attempts if row["id"] > offset)
            if reporting
            else {}
        )
        for group in groups:
            logger.info(
                "Finished batch %s",
                group,
                extra={
                    "progress": dict(kind="batch_finished", batch_id=str(group), status=outcome)
                },
            )
        logger.info(
            "Search %s",
            outcome,
            extra={"progress": dict(kind="search_finished", status=outcome, reason=outcome)},
        )
        if reporting:
            generator._streaming_progress = None
