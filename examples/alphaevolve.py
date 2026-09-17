"""Evolve a CartPole policy using a Slick API provider and Docker evaluation."""

import argparse
import asyncio
from functools import partial
from pathlib import Path

from slick import prompts
from slick.providers import OpenAIAPI

import rsikit.alphaevolve as alphaevolve
from rsikit.alphaevolve import AlphaEvolve, evaluate_program

INITIAL = """from rsikit import Controller

class Solution(Controller):
    async def act(self, observation):
        return 0
"""


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/cartpole/solution.py"))
    args = parser.parse_args()
    # Slick's root is configured once by the application, never by library imports.
    prompts.TEMPLATE_ROOT = Path(alphaevolve.__file__).parent / "prompts"
    agent = AlphaEvolve(
        task="Balance the CartPole pole for as many steps as possible.",
        context=(
            "CartPole-v1: observation is [cart position, cart velocity, pole angle, "
            "pole angular velocity]. Action 0 pushes left; 1 pushes right. "
            "Each surviving step earns 1 reward. Evaluation averages seeds 0, 1, 2 "
            "with a 200-step cap. Optimize the reward metric."
        ),
        provider=OpenAIAPI(model=args.model, max_output_tokens=4096),
        evaluate=partial(evaluate_program, make_env="CartPole-v1", seeds=(0, 1, 2), max_steps=200),
    )
    best = await agent.run(INITIAL, attempts=args.attempts, concurrency=1, seed=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(best.content, encoding="utf-8")
    print(f"Best metrics: {best.metrics}; saved {args.output}")
    print(f"Generation calls: {agent.generation_calls}; evaluations: {agent.evaluations}")


if __name__ == "__main__":
    asyncio.run(main())
