"""Compose samplers, mutation/crossover, archives and islands on an offline packing task."""

import argparse
import asyncio
import json
import random
import statistics
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from slick import prompts
from slick.providers import Provider

from rsikit import Candidate, Draft, Evaluation, PromptProposer, SequentialStrategy
from rsikit.archives import EliteArchive, FeatureGrid, QDArchive
from rsikit.examples.circle_packing.evaluate import evaluate, read_circles
from rsikit.examples.prompt_search import PACKING, BudgetedProvider
from rsikit.islands import Archive, migrate
from rsikit.selection import UCB1, ThompsonSampling, tournament


class ConceptSearch(SequentialStrategy):
    """A concrete steady-state recipe: independent island/operator credit, optional migration.

    Binary success means valid child quality strictly exceeds its primary parent's.
    Rejections receive zero, infrastructure errors no reward; admission is independent.
    The experiment supplies measured seeds, task/provider via its proposer, evaluation,
    archives and policies. Model weights and the evaluator stay fixed.
    """

    def __init__(
        self,
        initial: Candidate,
        proposer: PromptProposer,
        evaluate_source: Callable[[str], Awaitable[Evaluation]],
        islands: Sequence[Archive],
        *,
        island_sampler: UCB1 | ThompsonSampling,
        operator_sampler: UCB1 | ThompsonSampling,
        select_parent: Callable[[tuple[Candidate, ...]], Candidate],
        objective: str = "sum_radii",
        maximize: bool = True,
    ):
        self.operations, self.evaluate_source = proposer, evaluate_source
        self.islands = islands
        self.island_sampler, self.operator_sampler = island_sampler, operator_sampler
        self.parent_selector = select_parent
        self.allocations: list[dict] = []
        self.migration: list[dict] = []

        async def propose(parent, history):
            return await self.operations(parent, history, context=self.context)

        super().__init__(
            initial.source, initial.evaluation, propose, objective=objective, maximize=maximize
        )
        for archive in islands:
            archive.add(self.best)

    def select_parent(self) -> Candidate:
        island = self.island_sampler.choose(tuple(range(len(self.islands))))
        population = self.islands[island].candidates
        eligible = ("mutate", "crossover") if len(population) >= 2 else ("mutate",)
        operation = self.operator_sampler.choose(eligible)
        primary = self.parent_selector(population)
        parents = (primary,)
        if operation == "crossover":
            parents += (self.parent_selector(tuple(c for c in population if c.id != primary.id)),)
        self.context = dict(island=island, operation=operation, parents=parents)
        self.allocations.append(
            dict(
                attempt_id=len(self.history),
                island=island,
                operation=operation,
                parents=[c.id for c in parents],
            )
        )
        return primary

    def observe(self, candidate: Candidate) -> None:
        admitted = self.islands[self.context["island"]].add(candidate)
        parent = self.context["parents"][0]
        reward = 0.0
        if candidate.evaluation.valid:
            delta = self._score(candidate.evaluation) - self._score(parent.evaluation)
            reward = float(delta > 0 if self.maximize else delta < 0)
        self.allocations[-1]["admitted"] = admitted
        self.credit(reward, "evaluated" if candidate.evaluation.valid else "rejected")

    def credit(self, reward: float | None, outcome: str) -> None:
        record = self.allocations[-1]
        baseline = self._score(self.context["parents"][0].evaluation)
        for sampler, arm in (
            (self.island_sampler, record["island"]),
            (self.operator_sampler, record["operation"]),
        ):
            sampler.observe(
                arm, reward, attempt_id=record["attempt_id"], outcome=outcome, baseline=baseline
            )
        record.update(reward=reward, outcome=outcome)

    async def run(self, attempts: int, *, migration_interval: int = 0) -> None:
        for step in range(attempts):
            before = len(self.allocations)
            try:
                candidates = await self.generate()
                evaluations = [await self.evaluate_source(c.source) for c in candidates]
                await self.update(candidates, evaluations)
            except asyncio.CancelledError:
                if len(self.allocations) > before and "outcome" not in self.allocations[-1]:
                    self.credit(None, "cancelled")
                raise
            except BaseException as exc:
                if len(self.allocations) > before and "outcome" not in self.allocations[-1]:
                    self.allocations[-1]["error"] = f"{type(exc).__name__}: {exc}"
                    self.credit(None, "error")
                raise
            if migration_interval and (step + 1) % migration_interval == 0:
                routes = [(i, (i + 1) % len(self.islands)) for i in range(len(self.islands))]
                self.migration.extend(
                    migrate(self.islands, routes, select=lambda population: population)
                )


async def measure(source: str) -> Evaluation:
    """Parse literal geometry only; never execute generated source."""
    try:
        circles = read_circles(source)
    except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        return Evaluation(valid=False, feedback=f"{type(exc).__name__}: {exc}")
    result = evaluate(circles)
    if result["valid"]:
        result["metrics"]["mean_x"] = statistics.fmean(x for x, _, _ in circles)
    return Evaluation(**result)


async def demo(provider: Provider, *, qd: bool = False) -> dict:
    initial = (PACKING / "initial.py").read_text()
    task = (Path(__file__).parent / "prompts/literal_packing.txt").read_text()
    grid = FeatureGrid(bounds=((0.0, 1.0),), bins=(2,))
    archives = (
        [
            QDArchive(
                lambda c: grid.locate((c.evaluation.metrics["mean_x"],)),
                objective="sum_radii",
                cell_count=grid.size,
            )
            for _ in range(2)
        ]
        if qd
        else [EliteArchive(4, objective="sum_radii")]
    )
    provider = BudgetedProvider(provider, max_calls=6)
    rng = random.Random(7)
    search = ConceptSearch(
        Candidate(id=0, source=initial, evaluation=await measure(initial)),
        PromptProposer(task, provider),
        measure,
        archives,
        island_sampler=UCB1(role="island", seed=2),
        operator_sampler=ThompsonSampling(role="operator", seed=3),
        select_parent=lambda pool: tournament(
            pool, key=lambda c: c.evaluation.metrics["sum_radii"], rng=rng
        ),
    )
    await search.run(6, migration_interval=2 if qd else 0)
    return dict(
        recipe="adaptive-qd-islands" if qd else "elite-population",
        calls=provider.calls,
        best=search.best.model_dump(mode="json"),
        history=[c.model_dump(mode="json") for c in search.history],
        allocations=search.allocations,
        migration=search.migration,
        proposals=search.operations.records,
        operator_feedback=search.operator_sampler.observations,
        island_feedback=search.island_sampler.observations,
        archives=[[c.id for c in archive.candidates] for archive in archives],
        coverage=[archive.coverage for archive in archives] if qd else None,
    )


def responses() -> list[Draft]:
    initial = (PACKING / "initial.py").read_text()
    drafts = []
    for radius, mirrored in (
        (0.11, True),
        (0.09, True),
        (0.12, True),
        (0.115, False),
        (0.2, False),
        (0.125, True),
    ):
        source = initial.replace("0.10", str(radius))
        if mirrored:
            source = source.replace("(0.125, 0.625", "(0.875, 0.625").replace(
                "(0.375, 0.625", "(0.625, 0.625"
            )
        drafts.append(
            Draft(description="Change literal radii and horizontal arrangement", source=source)
        )
    return drafts


if __name__ == "__main__":
    from rsikit.tests.providers import ScriptedProvider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/concept-modules.json"))
    args = parser.parse_args()
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
    results = [asyncio.run(demo(ScriptedProvider(responses()), qd=qd)) for qd in (False, True)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    for result in results:
        assert result["calls"] == 6 and result["best"]["evaluation"]["metrics"]["sum_radii"] == 1.25
        print(
            f"{result['recipe']}: {result['calls']} scripted calls; coverage={result['coverage']}"
        )
    print(f"Saved {args.output}")
