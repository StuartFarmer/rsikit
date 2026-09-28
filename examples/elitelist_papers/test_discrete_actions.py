"""Regressions for valid action representations crashing environment evaluation."""

import unittest

import numpy as np

from rsikit import Policy
from rsikit.evaluation import PolicyError, run_episode


class DiscreteActionTests(unittest.IsolatedAsyncioTestCase):
    async def test_scalar_arrays_match_integer_actions_in_toy_text(self):
        class IntegerPolicy(Policy):
            async def act(self, observation):
                return 0

        class ArrayPolicy(Policy):
            async def act(self, observation):
                return np.array(0, dtype=np.int64)

        for env in ("Taxi-v4", "FrozenLake-v1", "CliffWalking-v1"):
            with self.subTest(env=env):
                expected = await run_episode(env, IntegerPolicy, env_seed=0, max_steps=2)
                actual = await run_episode(env, ArrayPolicy, env_seed=0, max_steps=2)
                self.assertEqual(actual[:4], expected[:4])
                self.assertEqual(actual[4]["episode"]["r"], expected[4]["episode"]["r"])

    async def test_invalid_discrete_actions_are_rejected_before_conversion(self):
        for action in (np.array([0]), np.array(0.5), np.array(6), np.array(-1)):
            with self.subTest(action=action):
                class InvalidPolicy(Policy):
                    async def act(self, observation):
                        return action

                with self.assertRaisesRegex(PolicyError, "Action outside action_space"):
                    await run_episode("Taxi-v4", InvalidPolicy, env_seed=0, max_steps=1)


class BoxActionTests(unittest.IsolatedAsyncioTestCase):
    async def test_car_racing_lists_and_tuples_match_arrays(self):
        class ArrayPolicy(Policy):
            async def act(self, observation):
                return np.array([0.0, 0.5, 0.0], dtype=np.float32)

        expected = await run_episode("CarRacing-v3", ArrayPolicy, env_seed=0, max_steps=2)
        for action in ([0.0, 0.5, 0.0], (0.0, 0.5, 0.0)):
            with self.subTest(action=action):
                class SequencePolicy(Policy):
                    async def act(self, observation):
                        return action

                actual = await run_episode("CarRacing-v3", SequencePolicy, env_seed=0, max_steps=2)
                np.testing.assert_array_equal(actual[0], expected[0])
                self.assertEqual(actual[1:4], expected[1:4])
                self.assertEqual(actual[4]["episode"]["r"], expected[4]["episode"]["r"])

    async def test_invalid_box_actions_still_raise_policy_errors(self):
        for action in ([0.0], [0.0, 2.0, 0.0], [0.0, float("nan"), 0.0],
                       np.array([0.0, 0.5, 0.0], dtype=np.float64)):
            with self.subTest(action=action):
                class InvalidPolicy(Policy):
                    async def act(self, observation):
                        return action

                with self.assertRaisesRegex(PolicyError, "Action outside action_space"):
                    await run_episode("CarRacing-v3", InvalidPolicy, env_seed=0, max_steps=1)


if __name__ == "__main__":
    unittest.main()
