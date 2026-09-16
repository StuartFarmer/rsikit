"""Save generation plots and a fixed-axis GIF from a packing run's history."""

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from PIL import Image

from rsikit import Candidate

from .evaluate import read_circles


def make_frame(history: list[Candidate], index: int, *, model: str = "", packings=None) -> Figure:
    """Draw one generation using axis limits from the entire supplied history."""
    scores = [
        c.evaluation.metrics["sum_radii"] if c.evaluation.valid else float("nan") for c in history
    ]
    feasible = [score for score in scores if score == score]
    padding = max((max(feasible) - min(feasible)) * 0.12, max(feasible) * 0.025)
    best = history[0]
    best_scores = []
    for candidate in history[: index + 1]:
        if candidate.evaluation.valid and (
            candidate.evaluation.metrics["sum_radii"] > best.evaluation.metrics["sum_radii"]
        ):
            best = candidate
        best_scores.append(best.evaluation.metrics["sum_radii"])

    candidate = history[index]
    valid = candidate.evaluation.valid
    shown = candidate if valid else best
    score = shown.evaluation.metrics["sum_radii"]
    status = (
        "Baseline"
        if index == 0
        else ("New best" if valid and candidate.id == best.id else "No new best")
    )
    if not valid:
        status = "Rejected — showing best packing"

    figure = Figure(figsize=(7, 9), dpi=110, facecolor="#f8fafc")
    packing, graph = figure.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
    figure.subplots_adjust(left=0.14, right=0.94, bottom=0.10, top=0.86, hspace=0.48)
    figure.suptitle(
        "Circle packing · 10 circles in the unit square", fontsize=15, fontweight="bold", y=0.97
    )
    figure.text(0.5, 0.932, f"Generation {index} / {len(history) - 1}", ha="center", fontsize=12)
    figure.text(0.5, 0.025, model, ha="center", fontsize=9, color="#475569")
    packing.set_title(
        f"{status}\nDisplayed score: {score:.6f}   |   Best so far: {best_scores[-1]:.6f}",
        fontsize=11,
        pad=12,
    )
    circles = read_circles(shown.source) if packings is None else packings[str(shown.id)]
    for circle_id, (x, y, radius) in enumerate(circles):
        packing.add_patch(
            Circle(
                (x, y), radius, facecolor="#93c5fd", edgecolor="#1d4ed8", linewidth=1.4, alpha=0.8
            )
        )
        packing.text(x, y, str(circle_id), ha="center", va="center", fontsize=9, color="#1e3a8a")
    packing.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal")
    packing.set_facecolor("white")

    generations = list(range(index + 1))
    graph.plot(
        generations,
        scores[: index + 1],
        "o-",
        color="#2563eb",
        label="Candidate",
        linewidth=2,
        markersize=5,
    )
    graph.plot(generations, best_scores, "--", color="#d97706", label="Best so far", linewidth=1.8)
    graph.axvline(index, color="#94a3b8", linewidth=1, linestyle=":")
    graph.set(
        xlim=(-0.15, max(1, len(history) - 1) + 0.15),
        ylim=(min(feasible) - padding, max(feasible) + padding),
        xlabel="Generation",
        ylabel="Sum of radii",
    )
    graph.xaxis.set_major_locator(MaxNLocator(integer=True))
    graph.ticklabel_format(axis="y", style="plain", useOffset=False)
    graph.grid(alpha=0.2)
    graph.legend(loc="lower right", fontsize=8)
    graph.set_title("Optimization progress · higher is better", fontsize=10)
    return figure


def render_run(directory: Path):
    """Rebuild all SVGs with final score limits and create a looping animation."""
    history = [
        Candidate.model_validate(c) for c in json.loads((directory / "history.json").read_text())
    ]
    summary = json.loads((directory / "summary.json").read_text())
    path = directory / "packings.json"
    packings = json.loads(path.read_text()) if path.exists() else None
    folder = directory / "generations"
    folder.mkdir(exist_ok=True)
    # ponytail: one image per generation in memory; stream video for very long runs.
    frames = []
    for index in range(len(history)):
        label = " · ".join(filter(None, (summary.get("strategy"), summary.get("model"))))
        figure = make_frame(history, index, model=label, packings=packings)
        try:
            figure.savefig(folder / f"{index:04d}.svg")
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
            frames.append(
                Image.frombytes(
                    "RGBA", canvas.get_width_height(), canvas.buffer_rgba().tobytes()
                ).convert("RGB")
            )
        finally:
            figure.clear()
    try:
        frames[0].save(
            directory / "progress.gif",
            save_all=True,
            append_images=frames[1:],
            duration=[1000] * (len(frames) - 1) + [2500],
            loop=0,
            disposal=2,
        )
    finally:
        for frame in frames:
            frame.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="existing run with history.json and summary.json"
    )
    args = parser.parse_args()
    render_run(args.directory)
    print(f"Animation: {args.directory / 'progress.gif'}")
