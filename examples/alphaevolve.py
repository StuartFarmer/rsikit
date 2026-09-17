"""Generate, evaluate, and update CartPole policies through OpenRouter."""

import argparse
import asyncio
import os
from pathlib import Path

import gymnasium as gym
from slick import prompts
from slick.providers import OpenRouterAPI

import rsikit.alphaevolve as alphaevolve
from rsikit import Executor, Run
from rsikit.alphaevolve import AlphaEvolve


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-oss-120b:nitro")
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running this example")

    prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent / "prompts"
    generator = AlphaEvolve(
        task="Balance the CartPole pole for as many steps as possible.",
        context=(
            "CartPole-v1: observation is [cart position, cart velocity, pole angle, "
            "pole angular velocity]. Action 0 pushes left; 1 pushes right. "
            f"Each surviving step earns 1 reward, up to {args.max_steps} steps."
        ),
        provider=OpenRouterAPI(model=args.model, max_output_tokens=8192, timeout=120),
    )
    executor = Executor(concurrency=args.concurrency)
    with (
        gym.make("CartPole-v1", max_episode_steps=args.max_steps) as environment,
        Run.create(
            name="cartpole-evolution", environment=environment, executor=executor, path=args.output
        ) as run,
    ):
        print(f"Run: {run.path}", flush=True)
        for generation in range(args.generations):
            policies = await generator.generate(n=args.batch_size)
            scores = await run.evaluate(policies)
            generator.update(scores)
            for policy in policies:
                print(
                    f"Generation {generation + 1}: {policy.name}: {scores[policy.id]:.1f}",
                    flush=True,
                )
            print(f"Best so far: {generator.best.name}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
