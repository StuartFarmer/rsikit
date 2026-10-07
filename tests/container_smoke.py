"""Paid-call-free launcher check: create, resume, or interrupt a mounted run."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from slick import prompts

import rsikit.generation as generation
from examples.inner_loop import evaluate_policies
from research.rollouts import Rollouts
from rsikit import Executor, Job, PolicyDefinition, Run, generate
from tests.providers import ScriptedProvider
from tests.test_execution import SOURCE, ProcessEnv


async def main(path, mode):
    console = Console()
    prompts.TEMPLATE_ROOT = Path(generation.__file__).parent / "prompts"
    provider = ScriptedProvider(
        [
            json.dumps(
                dict(
                    name="smoke",
                    description="Reviewed fixed policy",
                    implementation=SOURCE,
                )
            )
        ]
    )
    async with (
        Executor(episode_timeout=60) as executor,
        Run.open(path) if mode == "resume" else Run.create(name="smoke", path=path) as run,
    ):
        rollouts = Rollouts(ProcessEnv(), executor, run)
        if mode == "resume":
            (policy,) = run.policies()
            saved = run.path / "episodes" / policy.id / "0.pkl"
            original = saved.stat().st_mtime_ns
            with patch.object(executor, "_evaluate", wraps=executor._evaluate) as evaluation:
                await evaluate_policies(rollouts, [policy], [0, 1], resumed=True)
                assert [call.args[0].seed for call in evaluation.call_args_list] == [1]
            assert saved.stat().st_mtime_ns == original
            assert run.scores(policy) == {0: 2.0, 1: 2.0}
            console.print("[green]Resumed: cached seed 0; completed seed 1[/green]")
        else:
            policy = await generate("count", provider=provider)
            await evaluate_policies(rollouts, [policy], [0])
            run.save_policy(policy, scores={1: None})
            assert run.scores(policy) == {0: 2.0, 1: None}
            assert list((run.path / "exports").glob("*.py"))
            assert run.load_episode(policy, 0).artifacts["state.json"]
            (run.path / "provenance.json").write_text(
                json.dumps(
                    {
                        "image": os.environ.get("RSIKIT_IMAGE_ID"),
                        "generation_pid": os.getpid(),
                        "runtime_key_present": bool(os.environ.get("OPENAI_API_KEY")),
                    }
                )
            )
            console.print("[green]Generated, evaluated and saved[/green]")
            if mode == "cancel":
                source = SOURCE.replace("return 0", "while True: pass")
                console.print("Waiting for Ctrl-C", highlight=False)
                async for _ in executor.execute(
                    [Job(PolicyDefinition(source=source), ProcessEnv(), seed=2)]
                ):
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--mode", choices=("create", "resume", "cancel"), default="create")
    args = parser.parse_args()
    asyncio.run(main(args.path, args.mode))
