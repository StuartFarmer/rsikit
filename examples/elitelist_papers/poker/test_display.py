"""Pinned elite snapshots, shared optimizer logs, and nested progress."""

import io
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from research.elitesearch import Config, Generation, Organism

from .display import SearchDisplay


class DisplayTests(unittest.TestCase):
    def test_leaders_survive_rescoring_logs_scroll_above_and_handlers_restore(self):
        output = io.StringIO()
        console = Console(file=output, width=100, height=20, force_terminal=True, color_system=None)
        old = Organism(
            id=1, generation=1, kind="new", name="Leader [bold]", score=None, status="evaluating"
        )
        agent = SimpleNamespace(
            config=Config(population_size=2, elite_size=1, generations=3),
            organisms=[old, Organism(id=2, generation=2, kind="edit", status="generated")],
            elites=[old],
            generations=[Generation(number=1, status="completed"), Generation(number=2)],
            history=[{"generation": 1, "elite_ids": [1], "scores": {"1": 7.25}}],
        )
        logger = logging.getLogger("research.elitesearch")
        before = (logger.level, logger.propagate, list(logger.handlers))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            display = SearchDisplay(agent, console)
            with self.assertRaisesRegex(RuntimeError, "stop"), display.run(path):
                display.checkpoint()
                self.assertEqual(display.progress.tasks[display.steps].total, 2)
                logger.info(
                    "Generated contender — useful description", extra={"event": "policy_generated"}
                )
                display.tables(0, 4)
                display.tables(2, 4)
                snapshot = io.StringIO()
                Console(file=snapshot, width=100).print(display.render())
                text = snapshot.getvalue()
                self.assertIn("Leader [bold]", text)
                self.assertIn("7.250", text)
                self.assertLess(text.index("Current elites"), text.index("Generations"))
                self.assertIn("Elite slots filled", text)
                self.assertIn("Generation 2 phases", text)
                self.assertIn("Table blocks", text)
                self.assertIn("Generated contender", output.getvalue())
                self.assertIn("useful description", path.read_text())
                self.assertLessEqual(
                    len(console.render_lines(display.render(), console.options)), console.height
                )
                display.heldout(0, 1, "Leader [bold]")
                heldout = io.StringIO()
                Console(file=heldout, width=100).print(display.render())
                self.assertIn("7.250", heldout.getvalue())
                self.assertIn("Held-out winners", heldout.getvalue())
                raise RuntimeError("stop")
            self.assertEqual((logger.level, logger.propagate, logger.handlers), before)
            self.assertEqual(console._live_stack, [])

    def test_small_terminal_shows_top_elites_and_keeps_bars(self):
        console = Console(file=io.StringIO(), width=90, height=18)
        rows = [
            Organism(
                id=i, generation=1, kind="new", name=f"Leader {i}", score=i, status="evaluated"
            )
            for i in range(1, 21)
        ]
        agent = SimpleNamespace(
            config=Config(elite_size=20),
            organisms=rows,
            elites=rows,
            generations=[Generation(number=1, status="completed")],
            history=[
                {
                    "generation": 1,
                    "elite_ids": list(range(1, 21)),
                    "scores": {str(i): i for i in range(1, 21)},
                }
            ],
        )
        display = SearchDisplay(agent, console)
        display.checkpoint()
        rendered = display.render()
        self.assertLessEqual(len(console.render_lines(rendered, console.options)), console.height)
        output = io.StringIO()
        Console(file=output, width=90).print(rendered)
        self.assertIn("top", output.getvalue())
        self.assertIn("Elite slots filled", output.getvalue())


if __name__ == "__main__":
    unittest.main()
