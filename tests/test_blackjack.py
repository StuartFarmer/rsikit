"""Deterministic blackjack rules, hidden information, and runner integration."""

import asyncio
import unittest
from collections import Counter

import numpy as np
from gymnasium.error import InvalidAction
from gymnasium.utils.env_checker import check_env

from rsikit import envs
from rsikit.episode import PolicyError, run_episode
from rsikit.policy import Policy
from rsikit.sandbox.codec import decode_space, encode_space


class BlackjackTests(unittest.TestCase):
    def env(self, cards=(), **kwargs):
        self.assertTrue(hasattr(envs, "BlackjackEnv"), "finite-shoe environment is missing")
        env = envs.BlackjackEnv(**{"shoes_per_episode": 1, **kwargs})
        env.reset(seed=7)
        # Rig only the private shoe, never add a public API that exposes its order.
        env._shoe[: len(cards)] = cards
        return env

    def exposed(self, obs):
        return tuple(obs[10 : 10 + obs[9]])

    def test_rules_and_rewards(self):
        cases = [
            # Cards are dealt player, dealer upcard, player, dealer hole, then draws.
            ([1, 10, 10, 9], [], 1.5),
            ([1, 1, 10, 10], [], 0),
            ([10, 1, 9, 10], [], -1),
            ([10, 10, 8, 8], [0], 0),
            ([10, 10, 7, 8], [0], -1),
            ([10, 6, 8, 10, 10], [0], 1),
            ([5, 6, 6, 10, 10, 10], [2], 2),
            ([10, 6, 6, 10, 10], [1], -1),
            ([1, 10, 5, 8, 10, 5], [1, 1], 1),
            ([10, 1, 8, 6, 2], [0], 1),
        ]
        for cards, actions, expected in cases:
            with self.subTest(cards=cards):
                env = self.env(cards)
                obs, reward, done, _, info = env.step(0)
                for action in actions:
                    obs, reward, done, _, info = env.step(action)
                self.assertEqual(reward, expected)
                self.assertTrue(info["round_complete"])
                self.assertFalse(done)  # Hand boundaries preserve policy memory.
                self.assertEqual(obs[0], 0)
        env = self.env([10, 1, 8, 6, 2], hit_soft_17=True)
        env.step(0)
        self.assertEqual(env.step(0)[1], -1)
        env = self.env([1, 10, 10, 9])
        self.assertEqual(env.step(3)[1], 12)  # Eight-unit wager before dealing.

    def test_splits(self):
        env = self.env([8, 6, 8, 10, 3, 10, 2, 10, 10])
        env.step(0)
        obs, *_ = env.step(3)
        self.assertEqual((obs[1], obs[6], obs[7]), (11, 0, 2))
        self.assertEqual(self.exposed(obs), (3,))
        obs, *_ = env.step(2)  # Double first split; now second hand receives its card.
        self.assertEqual((obs[1], obs[6]), (10, 1))
        self.assertEqual(self.exposed(obs), (10, 2))
        self.assertEqual(env.step(2)[1], 4)
        env = self.env([1, 6, 1, 10, 10, 10, 10])
        env.step(0)
        obs, reward, *_ = env.step(3)
        self.assertEqual(reward, 2)  # Split aces get one card; 21 pays 1:1.
        self.assertEqual(self.exposed(obs), (10, 10, 10, 10))
        env = self.env([10, 10, 10, 10, 1, 1])
        env.step(0)
        self.assertEqual(env.step(3)[1], 2)  # Non-ace split 21 also pays only 1:1.
        env = self.env([8, 6, 8, 10, 8], max_split_hands=2, double_after_split=False)
        env.step(0)
        obs, *_ = env.step(3)
        self.assertEqual(tuple(obs[42:]), (1, 1, 0, 0, 0))
        for action in (2, 3):
            with self.assertRaises(InvalidAction):
                env.step(action)

    def test_hidden_information_and_observation_ownership(self):
        left = self.env([10, 6, 7, 2, 10, 10, 6, 7, 2])
        right = self.env([10, 6, 7, 9, 10])
        a, *_ = left.step(0)
        b, *_ = right.step(0)
        np.testing.assert_array_equal(a, b)  # Neither the hole nor future cards leak.
        self.assertEqual(self.exposed(a), (10, 6, 7))
        saved = a.copy()
        obs, reward, *_ = left.step(1)
        self.assertEqual(reward, -1)
        self.assertEqual(self.exposed(obs), (10,))  # Dealer hole stays hidden on bust.
        np.testing.assert_array_equal(a, saved)
        right.step(0)
        obs, *_ = left.step(0)
        self.assertEqual(obs[0], 1)

    def test_sitting_out_observes_a_round_without_a_wager(self):
        env = self.env([10, 6, 6, 10, 2, 10])
        self.assertIn(0, env.bet_sizes)
        obs, reward, done, truncated, info = env.step(env.bet_sizes.index(0))
        self.assertEqual((reward, done, truncated), (0.0, False, False))
        self.assertEqual(obs[0], 0)
        self.assertEqual(self.exposed(obs), (10, 6, 6, 2, 10, 10))
        self.assertTrue(info["sat_out"])
        self.assertTrue(info["round_complete"])
        self.assertEqual(env._position, 6)
        self.assertTrue(env.observation_space.contains(obs))
        # Sitting out is masked during an actual wagered hand.
        env = self.env([10, 6, 7, 10])
        obs, *_ = env.step(0)
        self.assertEqual(obs[46], 0)
        with self.assertRaises(InvalidAction):
            env.step(4)
        # A spectator cannot see the dealer hole when the stand-in player busts.
        env = self.env([10, 6, 6, 10, 10], bet_sizes=(0, 1))
        obs, reward, *_ = env.step(0)
        self.assertEqual(reward, 0)
        self.assertEqual(self.exposed(obs), (10, 6, 6, 10))
        for cards in ([1, 10, 10, 9], [10, 1, 9, 10]):
            env = self.env(cards, bet_sizes=(0,))
            self.assertEqual(env.step(0)[1], 0)  # Neither natural changes the bankroll.
        env = self.env(bet_sizes=(0,))
        done = False
        while not done:
            _, reward, done, _, info = env.step(0)
            self.assertEqual(reward, 0)
            self.assertTrue(info["round_complete"])
        self.assertGreater(info["rounds"], 1)

    def test_public_card_accounting_across_random_shoes(self):
        self.env()
        rng = np.random.default_rng(8)
        for decks in (1, 2, 6, 8):
            env = envs.BlackjackEnv(decks=decks, penetration=1.0, shoes_per_episode=1)
            for seed in range(25):
                obs, _ = env.reset(seed=seed)
                seen, unseen = Counter(), Counter()
                done = False
                while not done:
                    action = int(rng.choice(np.flatnonzero(obs[42:])))
                    obs, _, done, _, info = env.step(action)
                    seen.update(self.exposed(obs))
                    if info.get("round_complete"):
                        # A busted player's dealer hole is consumed but never shown.
                        if all(sum(hand) > 21 for hand in env._hands):
                            unseen.update([env._dealer[1]])
                        self.assertEqual(seen + unseen, Counter(env._shoe[: env._position]))
                    self.assertTrue(env.observation_space.contains(obs))
                self.assertGreaterEqual(env._position, env.cut_card)
                self.assertLessEqual(env._position, 52 * decks)

    def test_shoes_seeding_validation_and_gym_contract(self):
        env = self.env()
        check_env(env, skip_render_check=True)
        space = decode_space(encode_space(env.observation_space))
        for decks in (1, 2, 6, 8):
            a, b = self.env(decks=decks), self.env(decks=decks)
            self.assertEqual(
                Counter(a._shoe),
                Counter({**dict.fromkeys(range(1, 10), 4 * decks), 10: 16 * decks}),
            )
            self.assertEqual(a._shoe, b._shoe)
            obs, _ = a.reset(seed=3)
            b.reset(seed=3)
            rng = np.random.default_rng(12)
            done = False
            rounds = 0
            while not done:
                action = int(rng.choice(np.flatnonzero(obs[42:])))
                result = a.step(action)
                other = b.step(action)
                obs, _, done, truncated, info = result
                np.testing.assert_array_equal(obs, other[0])
                self.assertEqual(result[1:], other[1:])
                self.assertTrue(space.contains(obs))
                self.assertFalse(truncated)
                rounds += int(info.get("round_complete", False))
            self.assertGreater(rounds, 0)
            self.assertLessEqual(a._position, 52 * decks)
            with self.assertRaises(RuntimeError):
                a.step(0)
        for kwargs in (
            {"decks": 0},
            {"decks": 1.5},
            {"penetration": 0},
            {"penetration": float("nan")},
            {"bet_sizes": (-1,)},
            {"max_split_hands": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                envs.BlackjackEnv(**kwargs)
        env = self.env([10, 6, 7, 10])
        for action in (-1, 99, 0.5, True, np.array([0])):
            with self.subTest(action=action), self.assertRaises(InvalidAction):
                env.step(action)
        env.step(0)
        before = env._position
        with self.assertRaises(InvalidAction):
            env.step(3)  # Non-pair cannot split; rejection does not consume cards.
        self.assertEqual(before, env._position)

    def test_existing_episode_runner(self):
        self.env()

        class Player(Policy):
            async def act(self, obs):
                return int(obs[0] == 1 and obs[1] < 17)

        _, _, done, truncated, info = asyncio.run(
            run_episode(envs.BlackjackEnv, Player, env_seed=1, policy_seed=2)
        )
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertGreater(info["rounds"], 1)
        self.assertGreater(info["episode"]["l"], info["rounds"])

    def test_multiple_shoes_match_separate_shoes_without_resetting_episode(self):
        from examples.blackjack import choose_action

        with self.assertRaises(ValueError):
            envs.BlackjackEnv(shoes_per_episode=0)
        long = envs.BlackjackEnv(shoes_per_episode=3)
        single = envs.BlackjackEnv(shoes_per_episode=1)
        obs, _ = long.reset(seed=13)
        other, _ = single.reset(seed=13)
        self.assertEqual(obs[2], 1)  # Public fresh-shoe signal, not a count.
        total = expected = 0
        rounds = 0
        for shoe in range(1, 4):
            while True:
                action = choose_action(other)
                other, reward, done, _, info = single.step(action)
                obs, actual, finished, _, details = long.step(action)
                total += actual
                expected += reward
                rounds += int(info.get("round_complete", False))
                self.assertEqual(actual, reward)
                self.assertEqual(finished, done and shoe == 3)
                if done:
                    self.assertEqual(details["rounds"], rounds)
                    self.assertEqual(details["shoe_shuffled"], shoe < 3)
                    self.assertEqual(obs[2], int(shoe < 3))
                    self.assertEqual(tuple(obs[10:42]), tuple(other[10:42]))
                    if shoe < 3:
                        other, _ = single.reset()
                        self.assertEqual(long._shoe, single._shoe)
                    break
                np.testing.assert_array_equal(obs, other)
        self.assertEqual(total, expected)
        self.assertEqual(long._shoe_number, 3)
        with self.assertRaises(RuntimeError):
            long.step(0)

    def test_masked_action_reaches_the_policy_repair_boundary(self):
        class InvalidPlayer(Policy):
            async def act(self, observation):
                return 3  # Bet eight units, then incorrectly split every hand.

        with self.assertRaisesRegex(PolicyError, "legal-action mask"):
            asyncio.run(run_episode(envs.BlackjackEnv, InvalidPlayer, env_seed=0))

    def test_optimizer_environment_factory(self):
        import cloudpickle

        from examples.alphaevolve import FEATURE_BOUNDS, TASKS, make_environment

        self.assertIn("Blackjack", TASKS)
        self.assertIn("Blackjack", FEATURE_BOUNDS)
        with make_environment("Blackjack") as env:
            self.assertIsInstance(env.unwrapped, envs.BlackjackEnv)
            self.assertIn("zero wager sits out", env.instructions)
            restored = cloudpickle.loads(cloudpickle.dumps(env))
            obs, _ = restored.reset(seed=1)
            self.assertTrue(restored.observation_space.contains(obs))
            self.assertTrue(restored.step(4)[4]["sat_out"])
            restored.close()
        with make_environment("Blackjack", max_steps=1) as env:
            env.reset(seed=1)
            self.assertTrue(env.step(4)[3])
            self.assertIn("truncated after 1 steps", env.instructions)
        with self.assertRaisesRegex(ValueError, "render"):
            make_environment("Blackjack", render_mode="rgb_array")


if __name__ == "__main__":
    unittest.main()
