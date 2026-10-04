"""Local checkpoints for the meta runner; never execute serialized Python state."""

import fcntl
import json
import os
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def archive(path):
    number = 1
    while (target := path.with_name(f"{path.name}.interrupted-{number}")).exists():
        number += 1
    path.rename(target)
    return target


@contextmanager
def run_lock(path):
    with (path / ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This run is already active; stop it before resuming") from exc
        # The pre-resume runner did not hold a lock. Detect it before migrating its files.
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) == os.getpid():
                continue
            try:
                args = shlex.split(fields[1])
            except ValueError:
                continue
            if "research.meta_ocean" not in args or "--output" not in args:
                continue
            index = args.index("--output") + 1
            if index < len(args) and Path(args[index]).resolve() == path.resolve():
                raise ValueError(f"Run is still active (PID {fields[0]}); stop it before resuming")
        yield


def upgrade_candidates(path, *, upgrade="parser"):
    """Retain first-generation programs and usage; archive incompatible candidate scores."""
    checkpoint = path / "checkpoint.json"
    upgrade_name, reason = {
        "parser": (
            "upgrade.json",
            "Reevaluate candidates after source extraction and repair-contract upgrade",
        ),
        "repairs": ("upgrade-repairs.json", "Reevaluate candidates under host-enforced repairs"),
        "task-context": (
            "upgrade-task-context.json",
            "Reevaluate candidates with host-provided task context on every generation call",
        ),
    }[upgrade]
    if checkpoint.exists() and (
        upgrade == "parser"
        or ((path / upgrade_name).exists() and read_json(path / upgrade_name).get("completed"))
    ):
        return
    organisms, upgrade = path / "organisms.json", path / upgrade_name
    if not organisms.exists() and not upgrade.exists():
        return
    if not upgrade.exists():
        save_json(
            upgrade,
            dict(
                reason=reason,
                organisms=read_json(organisms),
                generations=read_json(path / "generations.json"),
            ),
        )
    previous = read_json(upgrade)
    rows, generations = previous["organisms"], previous["generations"]
    # An upgrade restarts selection with the saved first population, retaining all evidence.
    # Later generations depended on scores from the old protocol and cannot be mixed in.
    for name in (
        "development",
        "validation",
        "test",
        "organisms.json",
        "generations.json",
        "leaderboard.json",
        "selection.json",
        "winner.py",
        "summary.json",
        "checkpoint.json",
    ):
        item = path / name
        if item.exists():
            archive(item)
    rows = [row for row in rows if row["generation"] == 1]
    for row in rows:
        row.update(score=None, seed_scores={})
        if row["policy_id"]:
            row.update(status="generated", error=None)
    generations = generations[:1]
    for generation in generations:
        generation.update(status="running", elite_ids=[], promoted_ids=[], error=None)
    save_json(checkpoint, dict(organisms=rows, generations=generations))
    save_json(upgrade, dict(previous, completed=True))
