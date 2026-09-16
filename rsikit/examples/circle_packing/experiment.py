"""End-to-end packing experiment: propose, repair feasibility, measure, select."""

import asyncio
import json
import math
import shutil
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from slick import prompt

from rsikit import (
    AlphaEvolve,
    DGMArchive,
    EoH,
    Evaluation,
    EvaluationError,
    HillClimb,
    PromptProposer,
    ProposalRejected,
    Proposer,
    ReflectionMemory,
    RepairingProposer,
    ShinkaEvolve,
    ShinkaProposer,
)
from rsikit.sandbox import PythonSandbox

from .evaluate import evaluate
from .visualize import render_run


class Thought(BaseModel, extra="forbid"):
    description: str = Field(min_length=1)
    source: str = Field(min_length=1)


class PackingProposer(Proposer):
    def __init__(self, initial, provider, *, evaluation_timeout=10, packings=None):
        self.initial, self.provider = initial, provider
        self.evaluation_timeout = evaluation_timeout
        self.packings = {} if packings is None else packings
        self.thoughts: dict[int, str] = {0: "Initial grid packing"}
        self.diagnoses: dict[int, str] = {}

    async def __call__(self, parent, history, *, context=None):
        context = context or {"operation": "mutate"}
        operation = context["operation"]
        if operation == "mutate":
            return await self.generate(parent, history[-8:], provider=self.provider)
        if operation == "alphaevolve":
            outcomes = [
                {"id": c.id, "circles": self.packings.get(str(c.id))}
                for c in (parent, *context["inspirations"])
            ]
            return await self.evolve(
                parent, context["inspirations"], history[-8:], outcomes, provider=self.provider
            )
        if operation == "dgm-archive":
            diagnosis = await self.diagnose(parent, provider=self.provider)
            self.diagnoses[len(history)] = diagnosis
            return await self.modify(parent, diagnosis, provider=self.provider)
        operations = {
            "INIT": self.initialize,
            "E1": self.explore_diverse,
            "E2": self.explore_shared,
            "M1": self.modify_structure,
            "M2": self.tune_settings,
            "M3": self.simplify,
        }
        parents = [
            {"candidate": c.model_dump(mode="json"), "description": self.thoughts.get(c.id, "")}
            for c in context["parents"]
        ]
        try:
            thought = await operations[operation](parents, provider=self.provider)
        except ValidationError as exc:
            raise ProposalRejected("", f"Invalid EoH thought/source response: {exc}") from exc
        self.thoughts[len(history)] = thought.description
        return thought.source

    async def repair(self, source, evaluation):
        print(f"  Repair: {evaluation.feedback}", flush=True)
        return await self.fix(source, evaluation, provider=self.provider)

    @prompt(template="generate.j2")
    async def generate(self, parent, history, *, generated: str) -> str:
        return generated

    @prompt(template="repair.j2")
    async def fix(self, source, evaluation, *, generated: str) -> str:
        return generated

    @prompt(template="alphaevolve.j2")
    async def evolve(self, parent, inspirations, history, outcomes, *, generated: str) -> str:
        return generated

    @prompt(template="dgm_diagnose.j2")
    async def diagnose(self, parent, *, generated: str) -> str:
        return generated

    @prompt(template="dgm_modify.j2")
    async def modify(self, parent, diagnosis, *, generated: str) -> str:
        return generated

    @prompt(template="eoh_initialize.j2", output_type=Thought)
    async def initialize(self, parents, *, generated: Thought) -> Thought:
        return generated

    @prompt(template="eoh_diverse.j2", output_type=Thought)
    async def explore_diverse(self, parents, *, generated: Thought) -> Thought:
        return generated

    @prompt(template="eoh_shared.j2", output_type=Thought)
    async def explore_shared(self, parents, *, generated: Thought) -> Thought:
        return generated

    @prompt(template="eoh_structure.j2", output_type=Thought)
    async def modify_structure(self, parents, *, generated: Thought) -> Thought:
        return generated

    @prompt(template="eoh_settings.j2", output_type=Thought)
    async def tune_settings(self, parents, *, generated: Thought) -> Thought:
        return generated

    @prompt(template="eoh_simplify.j2", output_type=Thought)
    async def simplify(self, parents, *, generated: Thought) -> Thought:
        return generated


def write_svg(path: Path, circles: list):
    shapes = "\n".join(
        f'<circle cx="{x}" cy="{1 - y}" r="{r}" fill="#93c5fd" '
        'fill-opacity="0.6" stroke="#1d4ed8" stroke-width="0.002"/>'
        for x, y, r in circles
    )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-0.02 -0.02 1.04 1.04" '
        'width="600" height="600"><rect width="1" height="1" fill="white" '
        'stroke="black" stroke-width="0.003"/>' + shapes + "</svg>",
        encoding="utf-8",
    )


async def run(directory: Path, provider, *, evaluation_timeout=10, **settings):
    async with PythonSandbox(timeout=evaluation_timeout) as sandbox:
        return await _run(directory, provider, sandbox=sandbox, **settings)


async def _run(
    directory: Path,
    provider,
    *,
    sandbox,
    iterations=5,
    max_repairs=2,
    timeout=180,
    strategy_name="hillclimb",
    seed=0,
    population_size=4,
    islands=4,
    prompt_mode="legacy",
    instructions=None,
    reflect=False,
    model_settings=None,
    shinka_options=None,
    shinka_providers=(),
):
    if strategy_name == "shinkaevolve":
        prompt_mode = "modular"
    if prompt_mode != "modular" and (instructions or reflect):
        raise ValueError("Editable instructions and reflection require modular prompt mode")
    folder = Path(__file__).resolve().parent
    initial = (folder / "initial.py").read_text(encoding="utf-8")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    packings = {}
    checked = {}

    async def measure(source, location):
        location.mkdir(parents=True)
        program = location / "candidate.py"
        program.write_text(source, encoding="utf-8")
        output = await sandbox(source)
        (location / "execution.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        result = (
            Evaluation(valid=False, feedback=output["error"])
            if "error" in output
            else Evaluation.model_validate(evaluate(output["value"]))
        )
        (location / "evaluation.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        if result.valid:
            (location / "circles.json").write_text(json.dumps(output["value"]), encoding="utf-8")
        return result

    checks = 0

    async def check(source):
        nonlocal checks
        if strategy_name == "shinkaevolve":
            # Novelty filtering follows repairs; defer execution until it accepts the source.
            try:
                compile(source, "<candidate>", "exec")
            except SyntaxError as exc:
                return Evaluation(valid=False, feedback=f"SyntaxError: {exc}")
            return Evaluation(valid=True)
        if source in checked:
            result, _ = checked[source]
            return Evaluation(valid=result.valid, feedback=result.feedback)
        location = directory / "checks" / f"{checks:04d}"
        checks += 1
        result = await measure(source, location)
        checked[source] = (result, location)
        if result.valid:
            packings[str(len(strategy.history))] = json.loads(
                (location / "circles.json").read_text()
            )
        return Evaluation(valid=result.valid, feedback=result.feedback)

    task = (folder / "prompts/task.txt").read_text(encoding="utf-8")
    operations = (
        ShinkaProposer(task, provider, ensemble=shinka_providers, instructions=instructions)
        if strategy_name == "shinkaevolve"
        else PromptProposer(task, provider, instructions=instructions)
        if prompt_mode == "modular"
        else PackingProposer(
            initial, provider, evaluation_timeout=sandbox.timeout, packings=packings
        )
    )
    memory = ReflectionMemory(task, provider) if reflect else None

    async def propose(parent, history):
        context = dict(strategy.context)
        if prompt_mode == "modular":
            context["evidence"] = [
                {"id": c.id, "circles": packings.get(str(c.id))}
                for c in (*context.get("parents", (parent,)), *context.get("inspirations", ()))
            ]
            context["guidance"] = (*context.get("guidance", ()), *(memory.texts if memory else ()))
        return await operations(parent, history, context=context)

    proposer = RepairingProposer(propose, check, operations.repair, max_repairs=max_repairs)
    baseline = await measure(initial, directory / "candidates/0000")
    if not baseline.valid:
        raise EvaluationError(f"Initial packing failed: {baseline.feedback}")
    packings["0"] = json.loads((directory / "candidates/0000/circles.json").read_text())
    options = {
        "hillclimb": (HillClimb, {}),
        "alphaevolve": (
            AlphaEvolve,
            {
                "islands": islands,
                "seed": seed,
                "cell": lambda c: (
                    sum(r > 0.15 for _, _, r in packings[str(c.id)]),
                    sum(0.25 < x < 0.75 and 0.25 < y < 0.75 for x, y, _ in packings[str(c.id)]),
                ),
            },
        ),
        "eoh": (EoH, {"population_size": population_size, "seed": seed}),
        # Non-overlap gives pi*sum(r^2) <= 1; Cauchy bounds sum(r) <= sqrt(10/pi).
        "dgm-archive": (DGMArchive, {"score_bounds": (0, math.sqrt(10 / math.pi)), "seed": seed}),
    }
    if strategy_name == "shinkaevolve":
        options["shinkaevolve"] = (
            ShinkaEvolve,
            {
                "islands": islands,
                "seed": seed,
                "models": len(operations.models),
                "novelty": operations.assess_novelty,
                "reflect": operations.summarize,
                **(shinka_options or {}),
            },
        )
    strategy_class, settings = options[strategy_name]
    strategy = strategy_class(initial, baseline, proposer, objective="sum_radii", **settings)

    def save(status):
        best = strategy.best.evaluation.metrics["sum_radii"]
        start = baseline.metrics["sum_radii"]
        records = {
            "summary.json": {
                "status": status,
                "model": getattr(provider, "model", "scripted"),
                "model_settings": dict(model_settings or {}),
                "strategy": strategy_name,
                "prompt_mode": prompt_mode,
                "reflect": reflect,
                "instructions": dict(instructions or {}),
                "seed": seed,
                "population_size": population_size if strategy_name == "eoh" else None,
                "islands": islands if strategy_name in ("alphaevolve", "shinkaevolve") else None,
                "iterations": iterations,
                "max_repairs": max_repairs,
                "timeout": timeout,
                "evaluation_timeout": sandbox.timeout,
                "sandbox_image": sandbox.image,
                "completed_attempts": len(strategy.history) - 1,
                "baseline": start,
                "best": best,
                "gain": best - start,
                "improved": best > start,
                "best_id": strategy.best.id,
            },
            "history.json": [c.model_dump(mode="json") for c in strategy.history],
            "packings.json": packings,
            "selections.json": strategy.selections,
            "thoughts.json": operations.thoughts,
            "diagnoses.json": operations.diagnoses,
            "repairs.json": [
                [
                    {"source": a.source, "evaluation": a.evaluation.model_dump(mode="json")}
                    for a in result.attempts
                ]
                for result in proposer.history
            ],
        }
        if prompt_mode == "modular":
            records["proposal_records.json"] = operations.records
        if strategy_name == "shinkaevolve":
            records["shinka.json"] = {
                "proposals": strategy.proposals,
                "events": strategy.events,
                "scratchpad": strategy.scratchpad,
                "model_gains": [[str(g) for g in gains] for gains in strategy.model_gains],
                "models": [getattr(model, "model", "scripted") for model in operations.models],
                "novelty_enabled": strategy.embed is not None,
            }
        if memory:
            records["reflections.json"] = memory.records
            records["reflection_attempts.json"] = memory.attempts
        for name, data in records.items():
            (directory / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
        (directory / "best.py").write_text(strategy.best.source, encoding="utf-8")

    print(
        f"Baseline sum of radii: {baseline.metrics['sum_radii']:.9g}\nOutput: {directory}",
        flush=True,
    )
    status = "interrupted"
    save("running")
    try:
        for step in range(iterations):
            print(f"Generation {step + 1}/{iterations}...", flush=True)
            repair_runs = len(proposer.history)
            before = len(strategy.history)
            deadline = asyncio.get_running_loop().time() + timeout
            candidates = await asyncio.wait_for(strategy.generate(), timeout)
            evaluations = []
            for candidate in candidates:
                destination = directory / "candidates" / f"{candidate.id:04d}"
                if strategy_name == "shinkaevolve":
                    result = await asyncio.wait_for(
                        measure(candidate.source, destination),
                        max(0, deadline - asyncio.get_running_loop().time()),
                    )
                    if result.valid:
                        packings[str(candidate.id)] = json.loads(
                            (destination / "circles.json").read_text()
                        )
                else:
                    result, location = checked[candidate.source]
                    shutil.copytree(location, destination)
                evaluations.append(result)
            checked.clear()
            await strategy.update(candidates, evaluations)
            if memory:
                for completed in strategy.history[before:]:
                    await asyncio.wait_for(
                        memory.observe(
                            strategy.history[completed.parent_id],
                            completed,
                            objective=strategy.objective,
                            maximize=strategy.maximize,
                        ),
                        max(0, deadline - asyncio.get_running_loop().time()),
                    )
            last = strategy.history[-1].evaluation
            repairs = (
                len(proposer.history[-1].attempts) - 1 if len(proposer.history) > repair_runs else 0
            )
            print(
                f"  operation={strategy.context['operation']}, valid={last.valid}, score={last.metrics.get('sum_radii')}, "
                f"repairs={repairs}, best={strategy.best.evaluation.metrics['sum_radii']:.9g}",
                flush=True,
            )
            if not last.valid:
                print(f"  {last.feedback}", flush=True)
            save("running")
        status = "complete"
    finally:
        save(status)
        print("Rendering packing SVGs and animation...", flush=True)
        write_svg(directory / "initial.svg", packings["0"])
        write_svg(directory / "best.svg", packings[str(strategy.best.id)])
        render_run(directory)
    best = strategy.best.evaluation.metrics["sum_radii"]
    print(f"Final: {baseline.metrics['sum_radii']:.9g} -> {best:.9g}", flush=True)
    print(f"Packing: {directory / 'best.svg'}", flush=True)
    print(f"Animation: {directory / 'progress.gif'}", flush=True)
    return strategy
