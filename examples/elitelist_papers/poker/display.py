"""The optimizer's shared Rich logs with a pinned leaderboard and progress footer."""

import logging
from contextlib import closing, contextmanager

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from rsikit.progress import ProgressHandler


def make_progress(console=None):
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=4,
    )


def update_tables(progress, task, done, total):
    if done == 0:
        progress.reset(task, total=total, visible=True)
    progress.update(task, completed=done, total=total, visible=True)


class SearchDisplay:
    def __init__(self, agent, console):
        self.agent, self.console = agent, console
        self.progress = make_progress(console)
        self.generations = self.progress.add_task("Generations", total=agent.config.generations)
        self.steps = self.progress.add_task("  Generation phases", total=2)
        self.population = self.progress.add_task(
            "    Population evaluated/discarded", total=agent.config.population_size
        )
        self.handler = ProgressHandler(self.progress)
        self.progress.update(self.handler.generation, description="    Generating policies")
        self.progress.update(self.handler.evaluation, description="    Evaluating policies")
        self.table_task = self.progress.add_task("      Table blocks", total=None, visible=False)
        self.elites = self.progress.add_task("  Elite slots filled", total=agent.config.elite_size)
        self.generation = 0
        self.leader_generation = 0
        self.leaders = []
        self.live = Live(console=console, get_renderable=self.render, refresh_per_second=4)

    @contextmanager
    def run(self, path):
        loggers = [
            logging.getLogger(name) for name in ("research.elitesearch", "rsikit", __package__)
        ]
        settings = [(logger.level, logger.propagate) for logger in loggers]
        with (
            closing(logging.FileHandler(path, encoding="utf-8")) as log,
            self.live,
        ):
            log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            for logger in loggers:
                logger.addHandler(self.handler)
                logger.addHandler(log)
                logger.setLevel(logging.INFO)
                logger.propagate = False
            try:
                yield self
            finally:
                for logger, (level, propagate) in zip(loggers, settings):
                    logger.removeHandler(self.handler)
                    logger.removeHandler(log)
                    logger.setLevel(level)
                    logger.propagate = propagate
                self.handler.close()

    def checkpoint(self):
        agent, progress = self.agent, self.progress
        progress.update(
            self.generations, completed=sum(g.status == "completed" for g in agent.generations)
        )
        progress.update(self.elites, completed=len(agent.elites))
        if agent.generations:
            generation = agent.generations[-1]
            if generation.number != self.generation:
                for task in (
                    self.steps,
                    self.population,
                    self.handler.generation,
                    self.handler.evaluation,
                ):
                    progress.reset(task)
                progress.update(self.handler.generation, visible=False)
                progress.update(self.handler.evaluation, visible=False)
                progress.update(self.table_task, visible=False)
                self.generation = generation.number
            rows = [row for row in agent.organisms if row.generation == generation.number]
            done = sum(row.status in ("evaluated", "discarded") for row in rows)
            progress.update(self.population, completed=done)
            if generation.status == "completed":
                step, phase = 2, "complete"
            elif done == len(rows):
                step, phase = 2, "select elites"
            elif all(
                row.status in ("generated", "evaluating", "evaluated", "discarded") for row in rows
            ):
                step, phase = 1, getattr(agent, "phase", "tournament")
            else:
                step, phase = 0, "generate / repair"
                progress.update(self.table_task, visible=False)
            progress.update(
                self.steps,
                total=2,
                completed=step,
                description=f"  Generation {generation.number} phases — {phase}",
            )
            current_phase = getattr(agent, "phase", "")
            progress.update(
                self.handler.generation,
                description="    Repairing policies"
                if current_phase == "repair"
                else "    Generating policies",
            )
        if agent.history and agent.history[-1]["generation"] != self.leader_generation:
            snapshot = agent.history[-1]
            self.leader_generation = snapshot["generation"]
            self.leaders = [
                (
                    rank,
                    agent.organisms[row_id - 1].name,
                    snapshot["scores"][str(row_id)],
                    agent.organisms[row_id - 1].kind,
                    agent.organisms[row_id - 1].parent_ids,
                )
                for rank, row_id in enumerate(snapshot["elite_ids"], 1)
            ]
        if self.live.is_started:
            self.live.refresh()

    def tables(self, done, total):
        update_tables(self.progress, self.table_task, done, total)
        if self.live.is_started:
            self.live.refresh()

    def heldout(self, done, total, name):
        self.progress.update(
            self.steps, description="  Held-out winners", completed=done, total=total
        )
        self.progress.update(self.handler.generation, visible=False)
        self.progress.update(self.handler.evaluation, visible=False)
        if self.live.is_started:
            self.live.refresh()
        if name:
            logging.getLogger(__name__).info("Held-out evaluation: %s", name)

    def render(self):
        progress = self.progress.get_renderable()
        if not self.leaders:
            repairs = sum(row.repairs for row in self.agent.organisms)
            attempts = sum(
                r.get("phase") != "screen" for r in getattr(self.agent, "pending_reports", [])
            )
            board = Text(
                f"Current elites — waiting for first generation ({repairs} repairs; {attempts} tournament attempts)",
                style="dim",
            )
        else:
            # Keep the footer inside the viewport as elite capacity or terminal size changes.
            height = len(self.console.render_lines(progress, self.console.options))
            visible = max(1, self.console.height - height - 8)
            board = Table(
                title=f"Current elites — generation {self.leader_generation}", expand=True
            )
            for name in ("Rank", "Elite", "BB/100", "Origin", "Parents"):
                board.add_column(name, no_wrap=True, overflow="ellipsis")
            for rank, name, score, kind, parents in self.leaders[:visible]:
                board.add_row(
                    str(rank),
                    Text(" ".join(name.split())),
                    f"{score:.3f}",
                    kind,
                    ", ".join(map(str, parents)) or "—",
                )
            if visible < len(self.leaders):
                board.caption = (
                    f"Showing top {visible}/{len(self.leaders)}; full board in generation report"
                )
        return Group(board, progress)
