"""Benchmark completed blackjack rounds, including decisions and shoe resets.

Run: python -m examples.blackjack --hands 100000 --repeats 3 --compare-gym
This deliberately simple policy is a starting point for evolution, not basic strategy.
"""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import gymnasium as gym

from rsikit.policy import Policy


def choose_action(obs):
    if obs[0] == 0:
        return 0  # Fixed minimum wager; no built-in memory or strategy table.
    if obs[5] in (1, 8) and obs[45]:
        return 3
    if obs[1] == 11 and obs[44]:
        return 2
    return int(obs[1] < 17)


class Solution(Policy):
    async def act(self, observation):
        return choose_action(observation)


def benchmark(hands, repeats, decks, compare_gym=False):
    from rsikit.envs import BlackjackEnv

    results = []
    factories = [("finite_shoe", lambda: BlackjackEnv(decks=decks))]
    if compare_gym:
        factories.append(("gymnasium_blackjack_v1", lambda: gym.make("Blackjack-v1", natural=True)))
    for name, factory in factories:
        samples = []
        for repeat in range(repeats):
            env = factory()
            obs, _ = env.reset(seed=repeat)
            completed = steps = shoes = 0
            profit = 0.0
            start = perf_counter()
            while completed < hands:
                action = choose_action(obs) if name == "finite_shoe" else int(obs[0] < 17)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                profit += reward
                completed += int(
                    info.get("round_complete", False) if name == "finite_shoe" else done
                )
                if done or truncated:
                    obs, _ = env.reset()
                    shoes += 1
            elapsed = perf_counter() - start
            env.close()
            samples.append(
                {
                    "hands_per_second": hands / elapsed,
                    "seconds": elapsed,
                    "steps": steps,
                    "resets": shoes,
                    "profit": profit,
                }
            )
        results.append(
            {
                "environment": name,
                "median_hands_per_second": median(s["hands_per_second"] for s in samples),
                "samples": samples,
            }
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gymnasium": gym.__version__,
        "decks": decks,
        "hands_per_repeat": hands,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decks", type=int, default=6)
    parser.add_argument("--compare-gym", action="store_true")
    parser.add_argument("--min-hands-per-second", type=float, default=10000)
    args = parser.parse_args()
    if args.hands < 1 or args.repeats < 1:
        parser.error("hands and repeats must be positive")
    result = benchmark(args.hands, args.repeats, args.decks, args.compare_gym)
    print(json.dumps(result, indent=2))
    if result["results"][0]["median_hands_per_second"] < args.min_hands_per_second:
        raise SystemExit("Finite-shoe throughput below the requested threshold")


if __name__ == "__main__":
    main()
