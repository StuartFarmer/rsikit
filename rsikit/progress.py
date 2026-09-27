"""Common search progress events and policy score display."""

from statistics import fmean

from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text


class ProgressHandler(RichHandler):
    def __init__(self, progress, *, overlap=False):
        super().__init__(
            console=progress.console,
            show_path=False,
            markup=False,
            highlighter=None,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        self.progress = progress
        self.overlap = overlap
        self.generation = progress.add_task("Generating policies", total=0, visible=False)
        self.evaluation = progress.add_task("Evaluating policies", total=0, visible=False)

    def emit(self, record):
        event = getattr(record, "event", None)
        if event == "generation_started":
            if self.overlap:
                total = self.progress.tasks[self.generation].total + record.total
                self.progress.update(self.generation, total=total, visible=True)
            else:
                self.progress.update(self.evaluation, visible=False)
                self.progress.reset(self.generation, total=record.total, visible=True)
        elif event in ("policy_generated", "proposal_discarded"):
            self.progress.advance(self.generation)
        elif event == "evaluation_started":
            if self.overlap:
                total = self.progress.tasks[self.evaluation].total + record.total
                self.progress.update(self.evaluation, total=total, visible=True)
            else:
                self.progress.update(self.generation, visible=False)
                self.progress.reset(self.evaluation, total=record.total, visible=True)
        elif event in ("policy_evaluated", "evaluation_failed"):
            self.progress.advance(self.evaluation)
        super().emit(record)


def show_scores(policies, run, console, *, seeds=None):
    table = Table("Policy", "Description", "Score")
    seeds = None if seeds is None else tuple(dict.fromkeys(seeds))
    for policy in policies:
        scores = run.scores(policy)
        values = list(scores.values()) if seeds is None else [scores.get(seed) for seed in seeds]
        score = (
            f"{fmean(values):.1f}"
            if values and all(v is not None for v in values)
            else "unfinished"
        )
        table.add_row(Text(policy.name), Text(policy.description), score)
    console.print(table)
