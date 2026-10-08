"""SVG is the source frame, including for Gymnasium RGB and video export."""

import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from examples.blackjack import choose_action
from rsikit.envs import BitcoinEnv, BlackjackEnv
from rsikit.envs.bitcoin_render import BitcoinRenderer
from rsikit.envs.blackjack_render import BlackjackRenderer
from rsikit.envs.svg_frame import rasterize

NS = {"s": "http://www.w3.org/2000/svg"}


class SvgRenderTests(unittest.TestCase):
    def assert_vector_source(self, env):
        self.assertTrue(callable(getattr(type(env), "render_svg", None)))
        svg = env.render_svg()
        root = ET.fromstring(svg)
        self.assertEqual(root.attrib["viewBox"], "0 0 1280 720")
        self.assertEqual(root.findall(".//s:image", NS), [])
        self.assertTrue(root.findall(".//s:text", NS))
        self.assertIn("data:font/otf;base64,", svg)
        ids = [e.attrib["id"] for e in root.iter() if "id" in e.attrib]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("policy-name", ids)
        self.assertIn("score", ids)
        compact = env.render_svg(embed_fonts=False)
        self.assertNotIn("data:font", compact)
        self.assertIn("../fonts/", compact)
        np.testing.assert_array_equal(env.render(), rasterize(compact))
        np.testing.assert_array_equal(env.render(), rasterize(svg))
        self.assertEqual(svg, env.render_svg())
        return root

    def test_blackjack_source_hides_hole_card_and_preserves_split_hands(self):
        with BlackjackRenderer(BlackjackEnv(), policy_name='Policy <A> & "B"') as env:
            obs, _ = env.reset(seed=301)
            for _ in range(68):
                obs, *_ = env.step(choose_action(obs))
            root = self.assert_vector_source(env)
            self.assertEqual(root.find(".//s:text[@id='policy-name']", NS).text, 'Policy <A> & "B"')
            self.assertIsNotNone(root.find(".//s:g[@id='hand-1']", NS))
            self.assertIsNotNone(root.find(".//s:g[@id='hand-2']", NS))
            svg = env.render_svg()
            env.unwrapped._dealer[1] = env.unwrapped._shoe[0]
            self.assertEqual(svg, env.render_svg())

    def test_bitcoin_source_preserves_fills_and_dashed_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prices.csv"
            path.write_text("date,price_usd\n2020-01-01,100\n2020-01-02,200\n2020-01-03,100\n")
            with BitcoinRenderer(BitcoinEnv(path)) as env:
                env.reset(seed=7)
                for target in (1.0, 0.5):
                    env.step(np.array([target]))
                root = self.assert_vector_source(env)
                benchmark = root.find(".//s:polyline[@id='buy-hold']", NS)
                self.assertIsNotNone(benchmark)
                self.assertIn("stroke-dasharray", benchmark.attrib)
                self.assertIsNotNone(root.find(".//s:g[@id='fills']", NS))
                self.assertEqual([t["kind"] for t in env.trades], ["buy", "sell", "liquidation"])


if __name__ == "__main__":
    unittest.main()
