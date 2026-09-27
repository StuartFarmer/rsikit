"""Bounded table workers, equal exposure, and per-round population fitness."""

import asyncio
import logging
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter

from research.elitesearch import Measurement
from rsikit.episode import InfrastructureError

from .game import derived_seed, schedule
from .pool import TablePool
from .pool import docker_block as docker_block

CONTRACT = Path(__file__).with_name("CONTRACT.md").read_text()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TournamentConfig:
    rounds: int = 8
    deals: int = 100
    stack: int = 200
    workers: int = 4
    memory_gb: int = 32
    call_timeout: float = 2.0
    block_timeout: float = 180.0
    policy_ms: float = 5.0
    warmup_timeout: float = 15.0
    image: str = "elitetable-poker:local"


def interval(scores):
    """Descriptive bootstrap interval over complete independent scheduling rounds."""
    if len(scores) < 2:
        return None
    rng = random.Random(0)
    means = sorted(fmean(rng.choices(scores, k=len(scores))) for _ in range(1000))
    return [means[24], means[974]]


class Tournament:
    def __init__(self, config=TournamentConfig(), *, seed=0, runner=None, progress=None):
        self.config, self.seed, self.runner, self.progress = config, seed, runner, progress
        self.report = {}
        self.completed_blocks = {}
        self.cache_context = None

    async def __call__(self, policies):
        if self.runner is None:
            async with TablePool(self.config) as pool:
                return await self._evaluate(policies, pool)
        return await self._evaluate(policies, self.runner)

    async def _evaluate(self, policies, runner):
        config = self.config
        ids = [p.id for p in policies]
        if len(set(ids)) != len(ids):
            raise ValueError("A policy may occupy only one population slot")
        context = (config, self.seed)
        if self.cache_context != context:
            self.completed_blocks.clear()
            self.cache_context = context
        started = perf_counter()
        encounters, jobs = Counter(), []
        for round_index in range(config.rounds):
            groups = schedule(
                len(ids), derived_seed(self.seed, "schedule", round_index), encounters
            )
            for table, group in enumerate(groups):
                jobs.append((round_index, table, group))
                for a in group:
                    for b in group:
                        if a < b:
                            encounters[a, b] += 1
        slots = asyncio.Semaphore(config.workers)
        blocks, failures = [], {}
        active = {}
        reused, skipped = 0, 0
        if self.progress:
            self.progress(0, len(jobs))
        logger.info(
            "Tournament: %s policies, %s rounds, %s table blocks, %s workers; %s hands/block",
            len(ids),
            config.rounds,
            len(jobs),
            config.workers,
            config.deals * min(6, len(ids)),
        )

        async def play(round_index, table, group):
            nonlocal reused, skipped
            policy_ids = tuple(ids[i] for i in group)
            key = (round_index, table, policy_ids)
            request = dict(
                sources=[policies[i]._implementation for i in group],
                deals=config.deals,
                stack=config.stack,
                seed=derived_seed(self.seed, "cards", round_index, table),
                call_timeout=config.call_timeout,
                policy_ms=config.policy_ms,
                warmup_timeout=config.warmup_timeout,
                instructions=CONTRACT,
                label=f"Round {round_index + 1}, table {table + 1}",
                names=[f"{policies[i].name} [{ids[i][:8]}]" for i in group],
            )
            block_started = perf_counter()
            result = self.completed_blocks.get(key)
            cached = result is not None
            if result is not None:
                reused += 1
            else:
                async with slots:
                    if any(pid in failures for pid in policy_ids):
                        skipped += 1
                        return  # Retry only after these policies have been repaired.
                    block_started = perf_counter()
                    active[request["label"]] = block_started
                    logger.info("%s started: %s", request["label"], ", ".join(request["names"]))
                    try:
                        result = await runner(request, config)
                    finally:
                        active.pop(request["label"], None)
            if "failure" in result:
                index = result["failure"]
                if type(index) is not int or not 0 <= index < len(group):
                    raise InfrastructureError("Invalid failure attribution")
                failures[ids[group[index]]] = str(result["error"])
                logger.warning(
                    "%s failed after %.1fs: %s — %s",
                    request["label"],
                    perf_counter() - block_started,
                    request["names"][index],
                    result["error"],
                )
            else:
                if (
                    result.get("hands") != config.deals * len(group)
                    or len(result.get("chips", [])) != len(group)
                    or any(type(v) is not int for v in result["chips"])
                    or sum(result["chips"]) != 0
                    or any(
                        abs(v) > config.stack * (len(group) - 1) * result["hands"]
                        for v in result["chips"]
                    )
                ):
                    raise InfrastructureError("Invalid table chip totals or hand count")
                self.completed_blocks[key] = result
            blocks.append(
                dict(round=round_index, table=table, policies=[ids[i] for i in group], **result)
            )
            if self.progress:
                self.progress(len(blocks), len(jobs))
            logger.info(
                "Round %s, table %s: %s/%s blocks complete in %.1fs%s",
                round_index + 1,
                table + 1,
                len(blocks),
                len(jobs),
                perf_counter() - block_started,
                " (policy failure)" if "failure" in result else " (reused)" if cached else "",
            )

        tasks = [asyncio.create_task(play(*job)) for job in jobs]
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, timeout=5, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
                if not done:
                    logger.info(
                        "Tables running: %s; %s/%s blocks finished",
                        "; ".join(
                            f"{label} ({perf_counter() - since:.0f}s)"
                            for label, since in active.items()
                        ),
                        len(blocks),
                        len(jobs),
                    )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        blocks.sort(key=lambda block: (block["round"], block["table"]))
        self.report = dict(
            config=asdict(config),
            seed=self.seed,
            blocks=blocks,
            elapsed_seconds=perf_counter() - started,
            failures=failures,
            reused_blocks=reused,
            skipped_blocks=skipped,
        )
        if failures:
            logger.warning(
                "Repairing %s failed policies together; %s successful blocks retained, %s blocked tables skipped",
                len(failures),
                len(blocks) - sum("failure" in block for block in blocks),
                skipped,
            )
            # Do not rank an incomplete field. Unchanged table results remain reusable.
            return {pid: Measurement({}, failures.get(pid)) for pid in ids}
        chips = {pid: [0] * config.rounds for pid in ids}
        hands = {pid: [0] * config.rounds for pid in ids}
        for block in blocks:
            r = block["round"]
            for i, pid in enumerate(block["policies"]):
                chips[pid][r] += block["chips"][i]
                hands[pid][r] += block["hands"]
        measurements = {
            pid: Measurement({r: 50 * chips[pid][r] / hands[pid][r] for r in range(config.rounds)})
            for pid in ids
        }
        leaderboard = []
        for pid in ids:
            scores = list(measurements[pid].scores.values())
            positions, actions = Counter(), Counter()
            for block in blocks:
                if pid in block["policies"]:
                    index = block["policies"].index(pid)
                    positions.update(dict(enumerate(block["position_chips"][index])))
                    actions.update(block["actions"][index])
            leaderboard.append(
                dict(
                    policy_id=pid,
                    bb_per_100=fmean(scores),
                    hands=sum(hands[pid]),
                    ci95=interval(scores),
                    position_chips=dict(positions),
                    actions=dict(actions),
                )
            )
        self.report["leaderboard"] = sorted(
            leaderboard, key=lambda row: (-row["bb_per_100"], row["policy_id"])
        )
        self.report["table_hands"] = sum(block["hands"] for block in blocks)
        self.report["played_table_hands"] = (len(blocks) - reused) * config.deals * min(6, len(ids))
        self.report["hands_per_second"] = (
            self.report["played_table_hands"] / self.report["elapsed_seconds"]
        )
        return measurements
