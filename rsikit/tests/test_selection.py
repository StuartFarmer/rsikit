"""Selections use measured reward, preserve eligibility and isolate learning roles."""

import math
import random
import unittest

from rsikit import selection


class SelectionTests(unittest.TestCase):
    def test_direct_policies_respect_scores_and_eligibility(self):
        rng = random.Random(9)
        values = {"weak": -2.0, "best": 4.0}
        options = tuple(values)
        self.assertEqual(selection.uniform(("only",), rng=rng), "only")
        self.assertEqual(selection.weighted(options, (0, 1), rng=rng), "best")
        self.assertEqual(selection.tournament(options, key=values.get, size=2, rng=rng), "best")
        self.assertEqual(
            selection.tournament(options, key=values.get, size=2, maximize=False, rng=rng), "weak"
        )
        self.assertEqual(
            selection.epsilon_greedy(options, key=values.get, epsilon=0, rng=rng), "best"
        )
        self.assertEqual(
            selection.softmax(options, key=lambda x: values[x] * 1000, rng=rng), "best"
        )
        self.assertEqual(
            selection.softmax(options, key=lambda x: values[x] * 1000, maximize=False, rng=rng),
            "weak",
        )
        self.assertEqual(
            selection.epsilon_greedy(("weak",), key=values.get, epsilon=1, rng=rng), "weak"
        )

    def test_ucb_cold_starts_measured_credit_and_role_isolation(self):
        operator = selection.UCB1(role="operator", seed=4)
        island = selection.UCB1(role="island", seed=4)
        first = operator.choose(("mutate", "crossover"))
        operator.observe(first, 1, attempt_id=0, baseline=0.2)
        second = operator.choose(("mutate", "crossover"))
        self.assertNotEqual(first, second)
        operator.observe(second, 0, attempt_id=1)
        self.assertEqual(operator.choose((first, second)), first)
        self.assertEqual(operator.choose((second,)), second)
        self.assertEqual(island.counts, {})
        for attempt in range(2, 102):
            chosen = operator.choose((first, second))
            operator.observe(chosen, float(chosen == first), attempt_id=attempt)
        self.assertGreater(operator.counts[first], 80)
        self.assertGreater(operator.counts[second], 1)  # uncertainty still explores
        before = dict(operator.counts)
        operator.observe(first, None, attempt_id=102, outcome="provider_error")
        self.assertEqual(operator.counts, before)
        self.assertEqual(operator.observations[-1]["role"], "operator")
        self.assertIsNone(operator.observations[-1]["reward"])
        for reward in (math.nan, math.inf, -1, 2):
            with self.assertRaises(ValueError):
                operator.observe(first, reward, attempt_id=103)
        self.assertEqual(operator.counts, before)
        self.assertEqual(len(operator.observations), 103)

    def test_thompson_binary_posterior_and_reproducibility(self):
        policy = selection.ThompsonSampling(role="operator", seed=12)
        twin = selection.ThompsonSampling(role="operator", seed=12)
        for attempt in range(40):
            for sampler in (policy, twin):
                sampler.observe("good", 1.0, attempt_id=2 * attempt)
                sampler.observe("bad", 0.0, attempt_id=2 * attempt + 1)
        choices = [policy.choose(("good", "bad")) for _ in range(100)]
        self.assertEqual(choices, [twin.choose(("good", "bad")) for _ in range(100)])
        self.assertGreater(choices.count("good"), 95)
        self.assertEqual(policy.choose(("new",)), "new")
        self.assertEqual(policy.successes["good"], 40)
        self.assertEqual(policy.failures["bad"], 40)
        policy.observe("good", None, attempt_id=81, outcome="cancelled")
        with self.assertRaises(ValueError):
            policy.observe("good", 0.5, attempt_id=82)
        self.assertEqual(policy.successes["good"], 40)
        self.assertEqual(len(policy.observations), 81)

    def test_softmax_retains_probability_across_large_finite_score_range(self):
        rng = random.Random(6)
        choices = [
            selection.softmax((-1e308, 1e308), key=lambda value: value, temperature=1e308, rng=rng)
            for _ in range(200)
        ]
        self.assertGreater(choices.count(-1e308), 5)
        self.assertLess(choices.count(-1e308), 50)
