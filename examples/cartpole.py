"""A hand-written policy acting in unchanged Gymnasium CartPole."""

import asyncio

from rsikit.evaluation import PolicyError
from rsikit.policy import Policy


class Solution(Policy):
    async def act(self, observation):
        return int(observation[2] + 0.5 * observation[3] > 0)


async def main():
    from copy import deepcopy

    import gymnasium as gym

    from rsikit import evaluate

    with gym.make("CartPole-v1") as env:
        policy = Solution(deepcopy(env.observation_space), deepcopy(env.action_space))
        try:
            episode = await evaluate(policy, env, seed=1, max_steps=50)
        finally:
            await policy.close()
    if episode.error is not None:
        raise PolicyError(episode.error)
    print(
        f"reward={episode.total_reward} steps={len(episode)} "
        f"terminated={episode.terminations[-1]} truncated={episode.truncations[-1]}"
    )


if __name__ == "__main__":
    asyncio.run(main())
