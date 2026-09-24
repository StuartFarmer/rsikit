"""Offline model responses, real scoring and automatic per-generation video export."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from imageio_ffmpeg import count_frames_and_secs
from rich.console import Console

from tests.providers import ScriptedProvider
from tests.test_elitesearch import program


class BlackjackTrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_training_exports_each_completed_generation(self):
        from examples import blackjack_train, elitesearch

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
                patch.object(
                    elitesearch,
                    "OpenRouterAPI",
                    return_value=ScriptedProvider([program(0), program(1)]),
                ),
                patch.object(elitesearch, "Console", return_value=Console(file=io.StringIO())),
            ):
                await blackjack_train.main(
                    [
                        "--elites",
                        "1",
                        "--population",
                        "1",
                        "--generations",
                        "2",
                        "--new-fraction",
                        "1",
                        "--remix-fraction",
                        "0",
                        "--shoes-per-seed",
                        "1",
                        "--seeds",
                        "0",
                        "1",
                        "--heldout-seeds",
                        "99",
                        "--video-top",
                        "1",
                        "--video-workers",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            report = json.loads((output / "videos/manifest.json").read_text())
            self.assertEqual(report["seeds"], [0, 1])
            self.assertEqual(report["fps"], 30)
            self.assertEqual([r["generation"] for r in report["videos"]], [1, 2])
            for row in report["videos"]:
                self.assertEqual([s["seed"] for s in row["segments"]], [0, 1])
                self.assertEqual(
                    count_frames_and_secs(str(output / "videos" / row["video"]))[0], row["frames"]
                )
            log = (output / "run.log").read_text()
            self.assertLess(
                log.index("Queued leader videos for generation 1"), log.index("Generation 2:")
            )
            self.assertTrue((output / "videos/generation-01.log").is_file())
            self.assertTrue((output / "videos/generation-02.log").is_file())


if __name__ == "__main__":
    unittest.main()
