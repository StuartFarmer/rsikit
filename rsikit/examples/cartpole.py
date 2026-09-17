"""A hand-written policy acting in unchanged Gymnasium CartPole."""

import asyncio

import gymnasium as gym

from rsikit.policy import Policy

INSTRUCTIONS = (
    "Balance the pole on the cart. Observations are cart position, cart velocity, "
    "pole angle, and pole angular velocity. Action 0 pushes left; action 1 pushes right. "
    "Each step earns +1, including the final step. The task terminates if cart position "
    "exceeds +/-2.4 or pole angle exceeds +/-12 degrees. An external step limit truncates "
    "the episode. Keep the pole balanced for as many steps as possible."
)


class Solution(Policy):
    async def act(self, observation):
        return int(observation[2] + 0.5 * observation[3] > 0)


async def main():
    from rsikit.episode import run_episode

    episode = await run_episode(
        lambda: gym.make("CartPole-v1"),
        Solution,
        env_seed=1,
        policy_seed=2,
        max_steps=50,
        instructions=INSTRUCTIONS,
    )
    last = episode.transitions[-1] if episode.transitions else None
    print(
        f"status={episode.status} return={episode.return_} steps={episode.length} "
        f"terminated={last.terminated if last else None} "
        f"truncated={last.truncated if last else None}"
    )


if __name__ == "__main__":
    asyncio.run(main())
