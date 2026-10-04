"""Single-CSV trading, Bitcoin compatibility, and real search/evaluation wiring."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from gymnasium.error import InvalidAction
from gymnasium.utils.env_checker import check_env

from research.cli import parse_config
from research.experiment import evaluate_policies, run_experiment
from rsikit import envs
from tests.providers import ScriptedProvider

SOURCE = """import numpy as np
from rsikit import Policy
class Solution(Policy):
    async def act(self, observation):
        return np.array([1.0])
"""


class PriceSeriesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.assertTrue(hasattr(envs, "PriceSeriesEnv"), "PriceSeriesEnv is missing")
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.csv = self.root / "prices.csv"

    def test_csv_accounting_and_bitcoin_parity(self):
        self.csv.write_text("date,price_usd\n2020-01-01,100\n2020-01-02,200\n2020-01-03,100\n")
        generic = envs.PriceSeriesEnv(
            self.csv, price_column="price_usd", initial_cash=1000, fee_rate=0.01
        )
        bitcoin = envs.BitcoinEnv(self.csv, initial_cash=1000, fee_rate=0.01)
        check_env(generic, skip_render_check=True)
        for targets in ((1.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
            generic.reset()
            bitcoin.reset()
            for target in targets:
                a, reward, done, truncated, info = generic.step(np.array([target]))
                b, expected, end, trunc, old = bitcoin.step(np.array([target]))
                np.testing.assert_array_equal(a[:4], b[:4])
                self.assertEqual((reward, done, truncated), (expected, end, trunc))
                self.assertEqual(info["trade_value"], old["trade_usd"])
                self.assertEqual(info["liquidation_value"], old["liquidation_usd"])
            self.assertEqual(a[4], 2)
            if targets == (1.0, 0.0):
                self.assertAlmostEqual(info["wealth"], 1960.39603960396)

    async def test_no_trade_policy_is_rejected_even_when_trading_loses(self):
        self.csv.write_text("price\n100\n50\n25\n")
        for fee_rate in (0, 0.001):
            with self.subTest(fee_rate=fee_rate):
                env = envs.PriceSeriesEnv(self.csv, fee_rate=fee_rate)
                # Waiting is allowed; even a trade on the final bar qualifies.
                env.reset()
                self.assertFalse(env.step(np.array([0.0]))[2])
                self.assertTrue(env.step(np.array([1.0]))[2])
                # An earlier episode's trade must not qualify the next episode.
                env.reset()
                env.step(np.array([0.0]))
                with self.assertRaisesRegex(InvalidAction, "no trades"):
                    env.step(np.array([0.0]))

        hold, cash = self.root / "hold.py", self.root / "cash.py"
        hold.write_text(SOURCE)
        cash.write_text(SOURCE.replace("[1.0]", "[0.0]"))
        config = parse_config(
            [
                "evaluate",
                "--env",
                "PriceSeries",
                "--env-data-path",
                str(self.csv),
                "--env-fee-rate",
                "0",
                "--env-initial-cash",
                "1000",
                "--policy",
                str(hold),
                str(cash),
                "--output",
                str(self.root / "rejection"),
            ]
        )
        result = await evaluate_policies(config)
        measurements = list(result["measurements"].values())
        accepted = [m for m in measurements if m["accepted"]]
        rejected = [m for m in measurements if not m["accepted"]]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["scores"], {0: -750})
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["scores"], {})
        self.assertIn("no trades", rejected[0]["failure"])

    def test_irregular_intraday_ranges_and_causal_observations(self):
        self.csv.write_text(
            "timestamp,price\n2024-01-05T09:00:00Z,100\n2024-01-08T10:00:00+01:00,200\n2024-01-08T09:05:00Z,50\n"
        )
        env = envs.PriceSeriesEnv(self.csv, start_time="2024-01-06", fee_rate=0, initial_cash=1000)
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs, [200, 0, 0, 1, 0])
        result = env.step(np.array([1.0]))
        self.assertEqual(result[1:4], (-750, True, False))
        self.assertEqual(result[4]["timestamp"], "2024-01-08T09:05:00+00:00")
        np.testing.assert_array_equal(obs, [200, 0, 0, 1, 0])
        for last in (50, 500):
            self.csv.write_text(f"price\n100\n110\n{last}\n")
            env = envs.PriceSeriesEnv(self.csv, fee_rate=0, initial_cash=1000)
            np.testing.assert_array_equal(env.reset()[0], [100, 0, 0, 1, 0])
            np.testing.assert_allclose(env.step(np.array([0.5]))[0], [110, 0.1, 11 / 21, 1.05, 1])

    def test_invalid_csv_and_parameters_fail_before_trading(self):
        for content in (
            "price\n100\n",
            "close\n100\n200\n",
            "price,price\n100,100\n200,200\n",
            "price\n100\n0\n",
            "price\n100\n-2\n",
            "price\n100\nnan\n",
            "price\n100\ninf\n",
            "price\n100\nmissing\n",
            "price\n100,200\n300\n",
            "date,price\n2024-01-01,100\n2024-01-01,200\n",
            "date,price\n2024-01-02,100\n2024-01-01,200\n",
            "date,price\n2024-01-01,100\ninvalid,200\n",
            "date,price\n2024-01-01,100\n2024-01-02\n",
            'price\n100\n"200\n',
            'price\n100\n"2"00\n',
            '"price\n100\n200\n',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.csv.write_text(content)
                envs.PriceSeriesEnv(self.csv)
        self.csv.write_text("price\n100\n200\n")
        for options in (
            {"fee_rate": 1},
            {"initial_cash": 0},
            {"start_time": "2024-01-01"},
            {"time_column": "missing"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                envs.PriceSeriesEnv(self.csv, **options)

    async def test_single_csv_search_and_separate_period_evaluation(self):
        self.csv.write_text("when,close\n2024-01-05,100\n2024-01-08,200\n2024-01-09,50\n")
        config_path = self.root / "experiment.yaml"
        config_path.write_text(
            yaml.safe_dump(
                dict(
                    env="PriceSeries",
                    optimizer="elite",
                    model="scripted",
                    output="search",
                    environment=dict(
                        data_path="prices.csv",
                        price_column="close",
                        time_column="when",
                        asset_name="Example",
                        fee_rate=0,
                        initial_cash=1000,
                        end_time="2024-01-08",
                    ),
                    optimizer_options=dict(population=1, generations=1, elites=1, max_repairs=0),
                    budget=dict(spend_cap=1, input_price=0.01, output_price=0.01),
                )
            )
        )
        config = parse_config(["run", "--config", str(config_path)])
        self.assertEqual(config["environment"]["data_path"], str(self.csv))
        self.assertEqual(config["evaluation"]["seeds"], [0])
        raw = ScriptedProvider(
            [json.dumps(dict(name="Hold", description="Buy and hold", implementation=SOURCE))]
        )
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "scripted"}),
            patch("research.providers.UsageOpenRouter", return_value=raw),
        ):
            summary = await run_experiment(config)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue((self.root / "search/winner.py").is_file())
        dataset = json.loads((self.root / "search/dataset.json").read_text())
        self.assertEqual(dataset["sha256"], hashlib.sha256(self.csv.read_bytes()).hexdigest())
        self.assertEqual((dataset["start_row"], dataset["stop_row"]), (0, 2))
        self.assertEqual(
            json.loads((self.root / "search/leaderboard.json").read_text())[0]["score"], 1000
        )
        config = parse_config(
            [
                "evaluate",
                "--env",
                "PriceSeries",
                "--env-data-path",
                str(self.csv),
                "--env-price-column",
                "close",
                "--env-time-column",
                "when",
                "--env-start-time",
                "2024-01-08",
                "--env-fee-rate",
                "0",
                "--env-initial-cash",
                "1000",
                "--policy",
                str(self.root / "search/winner.py"),
                "--output",
                str(self.root / "evaluation"),
            ]
        )
        with patch(
            "research.providers.UsageOpenRouter", side_effect=AssertionError("No model needed")
        ):
            result = await evaluate_policies(config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(next(iter(result["measurements"].values()))["scores"], {0: -750})


if __name__ == "__main__":
    unittest.main()
