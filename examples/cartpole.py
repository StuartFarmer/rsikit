"""A hand-written policy acting in unchanged Gymnasium CartPole."""

import asyncio

from rsikit.controller import Controller


class Solution(Controller):
    async def act(self, observation):
        return int(observation[2] + 0.5 * observation[3] > 0)


async def main():
    from rsikit.episode import run_episode

    _, _, terminated, truncated, info = await run_episode(
        "CartPole-v1",
        Solution,
        env_seed=1,
        policy_seed=2,
        max_steps=50,
    )
    print(f"{info['episode']} terminated={terminated} truncated={truncated}")


if __name__ == "__main__":
    asyncio.run(main())
