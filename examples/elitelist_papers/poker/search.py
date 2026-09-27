"""EliteSearch breeding with a generation-wide, freshly rescored opponent field."""

import ast
import asyncio
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from research.elitesearch import EliteSearch, Generation, Organism
from rsikit.policy import _policy_class

logger = logging.getLogger(__name__)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class PokerSearch(EliteSearch):
    """Reuse generation/repair/prompts; population measurements replace point evaluations.

    Latest organism scores are mutable; history contains immutable generation
    snapshots. Only this class's save/load format supports these moving scores.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.target_score is not None:
            raise ValueError("Self-play scores cannot be used as a fixed early-stop target")
        self.history, self.pending_reports = [], []
        self.feedback = {}
        self.phase = "generate"

    async def _generate_all(self, rows, *, repairing=False):
        if rows:
            self.phase = "repair" if repairing else "generate"
            self._checkpoint()
            logger.info(
                "Generating %s policies%s",
                len(rows),
                " after repair" if repairing else "",
                extra={"event": "generation_started", "total": len(rows)},
            )

        async def produce(row):
            needs_repair = repairing or row.status in ("rejected", "execution_failed")
            if row.calls and row.calls[-1]["operation"] == "repair" and row.revisions:
                needs_repair = True
                if not row.implementation:
                    row.implementation = row.revisions[-1]["implementation"]
                    row.error = row.revisions[-1]["error"]
            await self._generate(row, repairing=needs_repair)

        tasks = [asyncio.create_task(produce(row)) for row in rows]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _experiment(self, rows):
        await self._generate_all(
            [
                row
                for row in rows
                if row.status != "discarded"
                and (row.id not in self._policies or row.status in ("rejected", "execution_failed"))
            ]
        )
        while True:
            field = self.elites + [row for row in rows if row.status != "discarded"]
            if len(field) < 2:
                raise RuntimeError("Fewer than two valid policies remain for self-play")
            for row in field:
                row.status, row.score, row.seed_scores = "evaluating", None, {}
            self._checkpoint()
            policies = [self._policies[row.id] for row in field]
            self.phase = "tournament"
            self._checkpoint()
            logger.info(
                "Evaluating %s policies in self-play (attempt %s)",
                len(field),
                1 + sum(r.get("phase") != "screen" for r in self.pending_reports),
                extra={"event": "evaluation_started", "total": len(field)},
            )
            results = await self.evaluate(policies)
            self.pending_reports.append(getattr(self.evaluate, "report", {}).copy())
            if set(results) != {row.policy_id for row in field}:
                raise ValueError("Evaluator must return exactly the field's policy IDs")
            failed = [row for row in field if results[row.policy_id].failure is not None]
            if not failed:
                panel = None
                for row in field:
                    scores = results[row.policy_id].scores
                    if (
                        not scores
                        or any(not math.isfinite(v) for v in scores.values())
                        or (panel is not None and set(scores) != panel)
                    ):
                        raise ValueError("Every player needs finite scores on the same round panel")
                    panel = set(scores)
                    row.score, row.seed_scores = (
                        fmean(scores.values()),
                        {str(k): v for k, v in scores.items()},
                    )
                    row.status, row.error = "evaluated", None
                    logger.info(
                        "Evaluated %s: score=%.3f BB/100",
                        row.name,
                        row.score,
                        extra={"event": "policy_evaluated"},
                    )
                return
            incumbents = {row.id for row in self.elites}
            repair = []
            for row in failed:
                row.error = results[row.policy_id].failure
                logger.warning(
                    "Evaluation failed for %s: %s",
                    row.name,
                    row.error,
                    extra={"event": "evaluation_failed"},
                )
                if row.id in incumbents:
                    # Preserve the incumbent's source and lineage; do not rewrite a past organism.
                    row.status = "discarded"
                    self.elites = [elite for elite in self.elites if elite.id != row.id]
                else:
                    row.status = "execution_failed"
                    repair.append(row)
            logger.warning(
                "Repairing/removing %s policies; completing the tournament with reusable table results",
                len(failed),
            )
            self._checkpoint()
            await self._generate_all(repair, repairing=True)
            # Rebuild scores from the complete current schedule; the evaluator reuses
            # only blocks with identical policies, seats and deals.

    def _promote(self, generation, rows):
        scores = {str(row.id): row.score for row in self.elites + rows if row.score is not None}
        super()._promote(generation, rows)
        self.history.append(
            dict(
                generation=generation.number,
                scores=scores,
                elite_ids=list(generation.elite_ids),
                attempts=self.pending_reports,
            )
        )
        report = self.pending_reports[-1] if self.pending_reports else {}
        self.feedback = {row["policy_id"]: row for row in report.get("leaderboard", [])}
        self.pending_reports = []

    def save(self, path):
        write_json(
            path,
            dict(
                version=1,
                config=asdict(self.config),
                task=self.task,
                context=self.context,
                rng_state=repr(self.rng.getstate()),
                reason=self.reason,
                organisms=[row.model_dump() for row in self.organisms],
                generations=[row.model_dump() for row in self.generations],
                elite_ids=[row.id for row in self.elites],
                history=self.history,
                pending_reports=self.pending_reports,
                feedback=self.feedback,
            ),
        )

    def load(self, path):
        data = json.loads(Path(path).read_text())
        if data["version"] != 1:
            raise ValueError("Unsupported poker checkpoint version")
        saved_config, config = data["config"].copy(), asdict(self.config)
        old_generations = saved_config.pop("generations")
        if config.pop("generations") < old_generations or saved_config != config:
            raise ValueError("Resume may only increase the generation budget")
        self.task, self.context = data["task"], data["context"]
        self.organisms = [Organism.model_validate(row) for row in data["organisms"]]
        self.generations = [Generation.model_validate(row) for row in data["generations"]]
        if [r.id for r in self.organisms] != list(range(1, len(self.organisms) + 1)):
            raise ValueError("Checkpoint organism IDs must be contiguous")
        self.elites = [self.organisms[i - 1] for i in data["elite_ids"]]
        self.history, self.pending_reports, self.feedback = (
            data["history"],
            data["pending_reports"],
            data["feedback"],
        )
        self.rng.setstate(ast.literal_eval(data["rng_state"]))
        for row in self.organisms:
            if row.policy_id:
                policy = _policy_class(row.name, row.implementation, row.description)
                if policy.id != row.policy_id:
                    raise ValueError("Checkpoint source does not match its policy ID")
                self._policies[row.id] = policy
            sources = [row.implementation] if row.policy_id else []
            sources.extend(r["implementation"] for r in row.revisions if r.get("policy_id"))
            for source in sources:
                self._sources.add(ast.dump(ast.parse(source), include_attributes=False))
