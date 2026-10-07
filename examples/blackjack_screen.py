"""Save a real split-hand frame for reviewing the blackjack table design."""

import argparse
from pathlib import Path

from PIL import Image

from examples.blackjack import choose_action
from rsikit.envs import BlackjackEnv
from rsikit.envs.blackjack_render import BlackjackRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--steps", type=int, default=68)
    parser.add_argument("--output", type=Path, default=Path("docs/blackjack-screen.png"))
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("steps must be nonnegative")
    with BlackjackRenderer(BlackjackEnv(), policy_name="Baseline agent") as env:
        obs, _ = env.reset(seed=args.seed)
        for _ in range(args.steps):
            obs, _, terminated, truncated, _ = env.step(choose_action(obs))
            if terminated or truncated:
                break
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".svg":
            args.output.write_text(env.render_svg(), encoding="utf-8")
        else:
            Image.fromarray(env.render()).save(args.output)
        print(
            f"{args.output}: seed={args.seed}, action={len(env.points) - 1}, net points={env.points[-1]:+g}"
        )


if __name__ == "__main__":
    main()
