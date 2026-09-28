"""Trade timing, fee-aware charts, and independent evaluation panels."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rsikit.envs import BitcoinEnv


class BitcoinRenderTests(unittest.TestCase):
    def test_renderer_tracks_fills_and_preserves_episode(self):
        from rsikit.envs.bitcoin_render import BitcoinRenderer

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prices.csv"
            path.write_text("date,price_usd\n2020-01-01,100\n2020-01-02,200\n2020-01-03,100\n")
            plain = BitcoinEnv(path, initial_cash=1000, fee_rate=0.01)
            env = BitcoinRenderer(BitcoinEnv(path, initial_cash=1000, fee_rate=0.01))
            obs, _ = plain.reset(seed=7)
            other, _ = env.reset(seed=7)
            np.testing.assert_array_equal(obs, other)
            first = env.render()
            self.assertEqual(first.shape, (720, 1280, 3))
            for target in (1.0, 0.5):
                expected = plain.step(np.array([target]))
                actual = env.step(np.array([target]))
                np.testing.assert_array_equal(expected[0], actual[0])
                self.assertEqual(expected[1:], actual[1:])
            self.assertEqual([t["index"] for t in env.trades], [0, 1, 2])
            self.assertEqual([t["kind"] for t in env.trades], ["buy", "sell", "liquidation"])
            self.assertEqual([t["price"] for t in env.trades], [100, 200, 100])
            self.assertAlmostEqual(env.history[-1]["equity"], actual[4]["wealth"])
            self.assertAlmostEqual(env.history[-1]["buy_hold"], 1000 * 0.99 / 1.01)
            self.assertGreater(env.history[-1]["drawdown"], 0)
            self.assertEqual(env.history[-1]["allocation"], 0)
            np.testing.assert_array_equal(env.render(), env.render())
            env.reset(seed=7)
            self.assertEqual(env.trades, [])
            self.assertEqual(len(env.history), 1)
            np.testing.assert_array_equal(first, env.render())
            for _ in range(2):
                env.step(np.array([0.0]))
            self.assertEqual(env.trades, [])
            self.assertEqual(env.history[-1]["equity"], 1000)

    def test_panels_use_unseen_dates_or_seeds(self):
        from examples.blackjack_videos import evaluation_panel

        experiment = dict(env="Bitcoin", seeds=[0], heldout_seeds=[99])
        env, seeds, panel = evaluation_panel(experiment, split="validation")
        train = BitcoinEnv()
        self.assertEqual(seeds, [99])
        self.assertEqual(env._dates[0], train._dates[-1])
        self.assertTrue(set(env._dates[1:]).isdisjoint(train._dates[1:]))
        self.assertEqual(panel["split"], "validation")
        section, _, panel = evaluation_panel(
            experiment, split="validation", start_date="2024-01-01", end_date="2024-01-03"
        )
        obs, _ = section.reset()
        self.assertEqual(len(section._prices), 3)
        self.assertEqual(obs[1:4].tolist(), [0, 0, 1])
        self.assertEqual(panel["start_date"], "2024-01-01")
        for start, end in (
            ("2024-01-01", "2024-01-01"),
            ("2010-01-01", None),
            (None, "2030-01-01"),
            ("2025-01-01", "2024-01-01"),
        ):
            with self.assertRaises(ValueError):
                evaluation_panel(experiment, split="validation", start_date=start, end_date=end)
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluation_panel(
                experiment, split="validation", data_path="rsikit/envs/data/bitcoin_train.csv"
            )
        _, seeds, _ = evaluation_panel(
            dict(env="Blackjack", seeds=[0], heldout_seeds=[99]), split="holdout"
        )
        self.assertEqual(seeds, [99])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluation_panel(dict(env="Blackjack", seeds=[0], heldout_seeds=[0]), split="holdout")

    def test_bitcoin_video_replays_trace_with_exact_frames(self):
        from imageio_ffmpeg import count_frames_and_secs

        from examples.blackjack_videos import render_policy

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "prices.csv"
            path.write_text("date,price_usd\n2020-01-01,100\n2020-01-02,200\n2020-01-03,100\n")
            env = BitcoinEnv(path)
            env.reset()
            reward = sum(env.step(np.array([target]))[1] for target in (1.0, 0.0))
            (root / "0.json").write_text(json.dumps(dict(actions=[[1.0], [0.0]], reward=reward)))
            result = render_policy(
                dict(
                    video=str(root / "test.mp4"),
                    env="Bitcoin",
                    data_path=str(path),
                    split="training",
                    seeds=[0],
                    traces=str(root),
                    policy_id="baseline",
                    name="Baseline",
                )
            )
            self.assertEqual(result["total_reward"], reward)
            self.assertEqual(result["frames"], 2)
            self.assertEqual(count_frames_and_secs(str(root / "test.mp4"))[0], 2)
            self.assertTrue((root / "test.png").exists())
            self.assertEqual(result["segments"][0]["final_equity"], 10000 + reward)


if __name__ == "__main__":
    unittest.main()
