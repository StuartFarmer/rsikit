"""Portfolio accounting, causal observations, data boundaries, and runner integration."""

import asyncio
import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from gymnasium.error import InvalidAction
from gymnasium.utils.env_checker import check_env

from rsikit import envs
from rsikit.policy import Policy
from tests.helpers import run_episode


class BitcoinTests(unittest.TestCase):
    def make_env(self, prices=(100, 200, 100), **kwargs):
        self.assertTrue(hasattr(envs, "BitcoinEnv"), "Bitcoin environment is missing")
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "prices.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["date", "price_usd"])
            writer.writerows(
                ((date(2020, 1, 1) + timedelta(days=i)).isoformat(), price)
                for i, price in enumerate(prices)
            )
        return envs.BitcoinEnv(data_path=path, initial_cash=1000, **kwargs)

    def test_fee_accounting_and_cash_objective(self):
        env = self.make_env(fee_rate=0.01)
        env.reset()
        obs, first, done, _, info = env.step(np.array([1.0]))
        self.assertFalse(done)
        # $1000 buys $990.099 worth of BTC plus $9.901 fee, then price doubles.
        self.assertAlmostEqual(info["wealth"], 1980.19801980198)
        self.assertEqual(obs[2], 1.0)
        obs, second, done, truncated, info = env.step(np.array([0.0]))
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["wealth"], 1960.39603960396)
        self.assertAlmostEqual(first + second, 960.39603960396)
        self.assertAlmostEqual(info["total_fees"], 29.70297029703)
        self.assertEqual(obs[2], 0.0)
        with self.assertRaises(RuntimeError):
            env.step(np.array([0.0]))
        env.reset()
        self.assertEqual(env.step(np.array([0.0]))[1], 0.0)
        self.assertEqual(env.step(np.array([0.0]))[4]["wealth"], 1000)
        env = self.make_env((100, 100), fee_rate=0.01)
        env.reset()
        # Forced terminal sale pays its own fee, even when the last action is buy.
        self.assertAlmostEqual(env.step(np.array([1.0]))[4]["wealth"], 980.19801980198)

    def test_partial_targets_drift_and_no_trade_fee(self):
        env = self.make_env((100, 100, 200, 200), fee_rate=0.01)
        env.reset()
        obs, _, _, _, info = env.step(np.array([0.5]))
        self.assertAlmostEqual(obs[2], 0.5)
        self.assertAlmostEqual(info["wealth"], 995.02487562189)
        obs, _, _, _, info = env.step(np.array([obs[2]]))
        self.assertAlmostEqual(obs[2], 2 / 3)  # Allocation drifts with the market.
        self.assertAlmostEqual(info["fee"], 0.0)
        # Flat final day: selling partially then liquidating costs the same as
        # selling all BTC at once: initial cash + doubled BTC value less 1%.
        _, _, _, _, info = env.step(np.array([0.25]))
        self.assertAlmostEqual(info["wealth"], 1482.587064676617)

    def test_no_future_leakage_and_observation_ownership(self):
        left = self.make_env((100, 110, 50), fee_rate=0)
        right = self.make_env((100, 110, 500), fee_rate=0)
        a, _ = left.reset()
        b, _ = right.reset()
        np.testing.assert_array_equal(a, b)
        original_observation = a
        a, *rest = left.step(np.array([0.5]))
        b, *other = right.step(np.array([0.5]))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(rest, other)
        np.testing.assert_array_equal(
            original_observation, [100, 0, 0, 1, date(2020, 1, 1).toordinal()]
        )
        # Mutating a returned observation cannot corrupt the price series.
        a[0] = -999
        self.assertGreater(left.step(np.array([0.0]))[0][0], 0)

    def test_validation_and_gym_contract(self):
        env = self.make_env()
        check_env(env, skip_render_check=True)
        env.reset()
        for action in ([-1], [1.01], [float("nan")], [float("inf")], [0, 1], [[0]], [True]):
            with self.subTest(action=action), self.assertRaises(InvalidAction):
                env.step(np.array(action))
        self.assertEqual(env.step(np.array([0.0]))[0][0], 200)
        for prices in ((0, 1), (1, float("nan")), (1,), (1, float("inf"))):
            with self.subTest(prices=prices), self.assertRaises(ValueError):
                self.make_env(prices)
        for rate in (-0.1, 1, float("nan")):
            with self.assertRaises(ValueError):
                self.make_env(fee_rate=rate)

    def test_download_split_and_runner(self):
        self.assertTrue(hasattr(envs, "BitcoinEnv"), "Bitcoin environment is missing")
        import cloudpickle

        from rsikit.envs.bitcoin import TRAIN_DATA, load_prices

        dates, prices = load_prices(TRAIN_DATA)
        validation_dates, _ = load_prices(Path("data/bitcoin/validation.csv"))
        self.assertEqual(dates[-1], validation_dates[0])
        self.assertEqual(date.fromordinal(dates[0]), date(2016, 9, 19))
        self.assertEqual(date.fromordinal(dates[-1]), date(2023, 9, 19))
        self.assertEqual(date.fromordinal(validation_dates[-1]), date(2026, 9, 18))
        self.assertEqual(len(dates), 2557)
        env = cloudpickle.loads(cloudpickle.dumps(envs.BitcoinEnv()))
        self.assertEqual(env.reset()[0][0], prices[0])

        class Cash(Policy):
            async def act(self, observation):
                return np.array([0.0])

        result = asyncio.run(run_episode(envs.BitcoinEnv, Cash, seed=7))
        self.assertEqual(result[4]["episode"]["r"], 0)
        self.assertEqual(result[4]["episode"]["l"], len(dates) - 1)

    def test_optimizer_factory_and_invalid_dates(self):
        from rsikit.envs.tasks import make_environment

        with make_environment("Bitcoin") as env:
            self.assertIsInstance(env, envs.BitcoinEnv)
            self.assertEqual(env.reset()[0][4], date(2016, 9, 19).toordinal())
        with self.assertRaises(ValueError):
            make_environment("Bitcoin", render_mode="rgb_array")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for day in ("2020-01-01", "2020-01-03", "2019-12-31"):
                path.write_text(f"date,price_usd\n2020-01-01,100\n{day},101\n")
                with self.assertRaises(ValueError):
                    envs.BitcoinEnv(data_path=path)

    def test_download_rejects_missing_days_without_overwriting(self):
        from unittest.mock import patch

        from examples import download_bitcoin

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.csv"
            source.write_text("asset,time,PriceUSD\nbtc,2016-09-19,100\n")
            with patch.object(download_bitcoin, "ROOT", root):
                with self.assertRaises(ValueError):
                    download_bitcoin.download(date(2026, 9, 19), source)
            self.assertFalse((root / "rsikit").exists())
        self.assertEqual(download_bitcoin.years_before(date(2024, 2, 29), 3), date(2021, 2, 28))


if __name__ == "__main__":
    unittest.main()
