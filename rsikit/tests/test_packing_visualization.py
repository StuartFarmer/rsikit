"""Check score ranges and honest handling of rejected packing generations."""

import unittest
from pathlib import Path

from rsikit import Candidate, Evaluation
from rsikit.examples.circle_packing.visualize import make_frame


class PackingVisualizationTests(unittest.TestCase):
    def test_shared_axes_include_future_best_and_keep_candidate_regression(self):
        source = (
            Path(__file__).resolve().parents[1] / "examples/circle_packing/initial.py"
        ).read_text()
        history = [
            Candidate(
                id=0, source=source, evaluation=Evaluation(valid=True, metrics={"sum_radii": 1})
            ),
            Candidate(
                id=1,
                source=source.replace("0.10", "0.125"),
                evaluation=Evaluation(valid=True, metrics={"sum_radii": 1.25}),
            ),
            Candidate(
                id=2,
                source=source.replace("0.10", "0.11"),
                evaluation=Evaluation(valid=True, metrics={"sum_radii": 1.1}),
            ),
            Candidate(id=3, source="broken", evaluation=Evaluation(valid=False)),
        ]
        figures = [make_frame(history, index) for index in range(len(history))]
        try:
            limits = [fig.axes[1].get_ylim() for fig in figures]
            self.assertTrue(all(limit == limits[0] for limit in limits))
            self.assertLess(limits[0][0], 1)
            self.assertGreater(limits[0][1], 1.25)
            self.assertEqual(list(figures[2].axes[1].lines[0].get_ydata()), [1, 1.25, 1.1])
            self.assertEqual(list(figures[2].axes[1].lines[1].get_ydata()), [1, 1.25, 1.25])
            self.assertAlmostEqual(figures[2].axes[0].patches[0].radius, 0.11)
            self.assertIn("Rejected", figures[3].axes[0].get_title())
            self.assertAlmostEqual(figures[3].axes[0].patches[0].radius, 0.125)
        finally:
            for figure in figures:
                figure.clear()

    def test_single_baseline_has_nonzero_axis_ranges(self):
        source = (
            Path(__file__).resolve().parents[1] / "examples/circle_packing/initial.py"
        ).read_text()
        history = [
            Candidate(
                id=0, source=source, evaluation=Evaluation(valid=True, metrics={"sum_radii": 1})
            )
        ]
        figure = make_frame(history, 0)
        try:
            lower, upper = figure.axes[1].get_ylim()
            self.assertLess(lower, 1)
            self.assertGreater(upper, 1)
        finally:
            figure.clear()
