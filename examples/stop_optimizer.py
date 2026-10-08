"""Run STOP from the host; generated improvers and policies execute in Docker."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from statistics import fmean

from slick import prompts
from slick.providers import OpenRouterAPI

from research import stop_optimizer
from research.stop_optimizer import STOP, ImproverExecutionError, Problem
from research.stop_optimizer.records import STOPEvent
from research.stop_optimizer.runtime import DockerRuntime
from rsikit import PolicyDefinition, Run
from rsikit.envs.tasks import TASKS


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", nargs=2, action="append", metavar=("ENV", "INITIAL_POLICY"), required=True
    )
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument("--image", default="rsikit:local")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--evaluations", type=int, default=4)
    parser.add_argument("--inner-generations", type=int, default=4)
    parser.add_argument("--inner-evaluations", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.seeds = list(dict.fromkeys(args.seeds))
    args.test_seeds = list(dict.fromkeys(args.test_seeds))
    if set(args.seeds) & set(args.test_seeds):
        parser.error("Development and final test seeds must be disjoint")
    if any(environment not in TASKS for environment, _ in args.problem):
        parser.error(f"Environments must be selected from {', '.join(TASKS)}")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")
    runtime = DockerRuntime(image=args.image, timeout=args.timeout)
    await runtime.preflight()
    prompts.TEMPLATE_ROOT = Path(stop_optimizer.__file__).parent / "prompts"
    provider = OpenRouterAPI(model=args.model, max_output_tokens=8192, timeout=120)
    with Run.create(name="stop-optimizer", path=args.output) as run:
        (run.path / "experiment.json").write_text(
            json.dumps(vars(args), default=str, indent=2) + "\n"
        )

        def record(kind, data):
            run.save(STOPEvent(kind=kind, data=data))

        problems = []
        baselines = []
        for index, (environment, filename) in enumerate(args.problem):
            initial = PolicyDefinition.from_file(filename)
            initial.validate()
            task_path = run.path / "problems" / str(index)
            report = await runtime.evaluate_policy(
                initial.to_text(),
                path=task_path,
                environment=environment,
                seeds=args.seeds,
                max_steps=args.max_steps,
            )
            if not report["accepted"] or not report["scores"]:
                raise ValueError(f"Invalid baseline for {environment}: {report.get('failure')}")
            baseline = fmean(report["scores"].values())
            baselines.append(
                dict(environment=environment, raw_mean=baseline, scores=report["scores"])
            )
            record("baseline", dict(index=index, **baselines[-1]))

            async def utility(
                source, *, environment=environment, path=task_path, baseline=baseline
            ):
                measured = await runtime.evaluate_policy(
                    source,
                    path=path,
                    environment=environment,
                    seeds=args.seeds,
                    max_steps=args.max_steps,
                )
                record("policy_measurement", dict(path=str(path), **measured))
                if not measured["accepted"] or not measured["scores"]:
                    return 0.0
                raw = fmean(measured["scores"].values())
                return min(
                    1 - 1e-9,
                    max(1e-9, 0.5 + math.atan((raw - baseline) / max(1, abs(baseline))) / math.pi),
                )

            instructions = report.get("instructions", TASKS[environment])
            problems.append(
                Problem(
                    initial.to_text(),
                    f"Produce Solution(Policy) source with async act(self, observation). Maximize cumulative reward. {instructions}",
                    utility,
                )
            )

        agent = STOP(
            "Improve policy search under explicit generation and evaluation budgets.",
            provider,
            runtime.execute,
            problems,
            on_event=record,
        )
        result = await agent.run(
            rounds=args.rounds,
            generations=args.generations,
            evaluations=args.evaluations,
            inner_generations=args.inner_generations,
            inner_evaluations=args.inner_evaluations,
        )
        (run.path / "improver.py").write_text(result["improver"])
        summary = dict(
            baselines=baselines,
            search_counts={
                name: result[name]
                for name in (
                    "generation_calls",
                    "utility_calls",
                    "meta_calls",
                    "downstream_checks",
                    "improver_checks",
                )
            },
            final_tests=[],
            final_test_evaluations=0,
        )
        for index, problem in enumerate(problems):
            caps = agent.capabilities(
                problem.evaluate,
                problem.utility_description,
                args.inner_generations,
                args.inner_evaluations,
            )
            try:
                source = await runtime.execute(result["improver"], problem.initial, caps)
            except ImproverExecutionError as exc:
                record(
                    "deployment_failed",
                    dict(problem=index, source=result["improver"], failure=str(exc)),
                )
                summary["final_tests"].append(dict(problem=index, failure=str(exc)))
                continue
            summary["final_test_evaluations"] += 1
            report = await runtime.evaluate_policy(
                source,
                path=run.path / "final_tests" / str(index),
                environment=args.problem[index][0],
                seeds=args.test_seeds,
                max_steps=args.max_steps,
            )
            record("final_test", dict(problem=index, **report))
            summary["final_tests"].append(dict(problem=index, **report))
            if report["accepted"]:
                PolicyDefinition.from_text(source).to_file(
                    run.path / "problems" / str(index) / "final-policy.py"
                )
        summary["counts_after_deployment"] = {
            name: getattr(agent, name)
            for name in (
                "generation_calls",
                "utility_calls",
                "meta_calls",
                "downstream_checks",
                "improver_checks",
            )
        }
        (run.path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(run.path)


if __name__ == "__main__":
    asyncio.run(main())
