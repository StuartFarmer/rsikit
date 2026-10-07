"""Price charts and saved-run replay use the original market and accounting."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from rsikit.envs import PriceSeriesEnv


class PriceSeriesRenderTests(unittest.IsolatedAsyncioTestCase):
    def test_renderer_preserves_trades_and_labels_irregular_or_undated_rows(self):
        from rsikit.envs.price_series_render import PriceSeriesRenderer

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prices.csv"
            for data, labels in (
                ("price\n100\n200\n100\n", ["0", "1", "2"]),
                (
                    "timestamp,price\n2024-01-05T09:00:00Z,100\n"
                    "2024-01-08T09:00:00Z,200\n2024-01-08T09:05:00Z,100\n",
                    ["2024-01-05 09:00:00", "2024-01-08 09:00:00", "2024-01-08 09:05:00"],
                ),
            ):
                with self.subTest(labels=labels):
                    path.write_text(data)
                    options = dict(initial_cash=1000, fee_rate=0.01, asset_name="EUR/USD")
                    plain = PriceSeriesEnv(path, **options)
                    env = PriceSeriesRenderer(PriceSeriesEnv(path, **options))
                    obs, _ = plain.reset(seed=7)
                    other, _ = env.reset(seed=7)
                    np.testing.assert_array_equal(obs, other)
                    initial = env.render()
                    for target in (1.0, 0.5):
                        expected = plain.step(np.array([target]))
                        actual = env.step(np.array([target]))
                        np.testing.assert_array_equal(expected[0], actual[0])
                        self.assertEqual(expected[1:], actual[1:])
                    self.assertEqual([p["date"] for p in env.history], labels)
                    self.assertEqual(
                        [t["kind"] for t in env.trades], ["buy", "sell", "liquidation"]
                    )
                    self.assertEqual([t["index"] for t in env.trades], [0, 1, 2])
                    self.assertAlmostEqual(env.history[-1]["buy_hold"], 1000 * 0.99 / 1.01)
                    self.assertEqual(env.render().shape, (720, 1280, 3))
                    np.testing.assert_array_equal(env.render(), env.render())
                    env.reset(seed=7)
                    np.testing.assert_array_equal(initial, env.render())
                    other_asset = PriceSeriesRenderer(PriceSeriesEnv(path, asset_name="GOLD"))
                    other_asset.reset(seed=7)
                    self.assertFalse(
                        np.array_equal(initial[:50, :500], other_asset.render()[:50, :500])
                    )

    async def test_export_saved_price_run_with_nondefault_accounting(self):
        from imageio_ffmpeg import count_frames_and_secs

        from research.elitesearch.videos import export

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "prices.csv"
            path.write_text("price\n1\n2\n3\n4\n")
            options = dict(data_path=str(path), asset_name="EUR/USD", initial_cash=1000, fee_rate=0)
            config = dict(
                env="PriceSeries",
                environment=options,
                evaluation=dict(seeds=[0], max_steps=None, timeout_per_seed=10),
                selection=dict(test_seeds=[]),
            )
            (root / "config.yaml").write_text(yaml.safe_dump(config))
            (root / "dataset.json").write_text(json.dumps(PriceSeriesEnv(**options).dataset))
            source = "import numpy as np\nfrom rsikit import Policy\nclass Solution(Policy):\n async def act(self, observation): return np.array([1.0])\n"
            with closing(sqlite3.connect(root / "run.sqlite")) as db:
                db.execute(
                    "CREATE TABLE elitesearch_organism (id, policy_id, name, implementation, seed_scores, score)"
                )
                db.execute(
                    "INSERT INTO elitesearch_organism VALUES (1, 'hold', 'Hold', ?, '{\"0\": 3000}', 3000)",
                    (source,),
                )
                db.execute("CREATE TABLE elitesearch_generation (number, status, elite_ids)")
                db.execute("INSERT INTO elitesearch_generation VALUES (1, 'completed', '[1]')")
                db.commit()
            await export(root, root / "videos", top=1, workers=1, generation=1)
            report = json.loads((root / "videos/manifest.json").read_text())
            video = report["videos"][0]
            self.assertEqual(video["total_reward"], 3000)
            self.assertEqual(video["segments"][0]["final_equity"], 4000)
            self.assertEqual(video["frames"], 3)
            mp4 = root / "videos" / video["video"]
            self.assertEqual(count_frames_and_secs(str(mp4))[0], 3)
            self.assertTrue(mp4.with_suffix(".png").exists())
            page = (root / "videos/index.html").read_text()
            self.assertIn("EUR/USD", page)
            self.assertIn("Final chart", page)
            self.assertNotIn("shoe(s)", page)
            # A theme update replaces rendered artifacts, without executing policies again.
            executions = (root / "videos/executions.jsonl").read_bytes()
            traces = {p: p.read_bytes() for p in (root / "videos/traces").rglob("*.json")}
            self.assertIn("render_version", report)
            cached_videos = set((root / "videos/policies").rglob("*.mp4"))
            with patch("research.elitesearch.videos.RENDER_VERSION", "test-theme-update"):
                await export(root, root / "videos", top=1, workers=1, generation=1)
                refreshed = json.loads((root / "videos/manifest.json").read_text())
                self.assertEqual(refreshed["render_version"], "test-theme-update")
                self.assertEqual(refreshed["videos"][0]["total_reward"], 3000)
                new_videos = set((root / "videos/policies").rglob("*.mp4")) - cached_videos
                self.assertEqual(len(new_videos), 1)
                rendered_at = {p: p.stat().st_mtime_ns for p in new_videos}
                await export(root, root / "videos", top=1, workers=1, generation=1)
                self.assertEqual(rendered_at, {p: p.stat().st_mtime_ns for p in new_videos})
            self.assertEqual((root / "videos/executions.jsonl").read_bytes(), executions)
            self.assertEqual(
                {p: p.read_bytes() for p in (root / "videos/traces").rglob("*.json")}, traces
            )
            # Replay can use a consistent host snapshot instead of the live database.
            snapshot = root / "snapshot.sqlite"
            (root / "run.sqlite").rename(snapshot)
            (root / "run.sqlite").write_bytes(b"unreadable live database")
            await export(root, root / "videos", top=1, workers=1, database=snapshot)
            self.assertEqual(
                json.loads((root / "videos/manifest.json").read_text())["videos"][0]["frames"], 3
            )
            path.write_text("price\n1\n2\n3\n5\n")
            with self.assertRaisesRegex(ValueError, "dataset differs"):
                await export(root, root / "videos", top=1, workers=1, generation=1)


if __name__ == "__main__":
    unittest.main()
