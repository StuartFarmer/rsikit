"""Evaluation records and a local script runner for trusted experiments."""

import asyncio
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, StrictBool, ValidationError

Metric = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class Evaluation(BaseModel, extra="forbid", frozen=True):
    """Correctness is a gate; named finite measurements determine selection."""

    valid: StrictBool
    metrics: dict[str, Metric] = Field(default_factory=dict)
    feedback: str = ""


class EvaluationError(RuntimeError):
    """The evaluator failed to supply its promised measurement contract."""


class LocalEvaluator:
    """Run a fixed script with --program PATH --output PATH on POSIX hosts.

    This is process separation, NOT a security sandbox. Candidate code inherits
    host permissions. Use only for trusted experiments; supply an isolated async
    evaluator for untrusted generated code. Scripts own datasets and validation.
    A timeout invalidates a candidate; script/protocol failures abort the search.
    """

    def __init__(self, script: Path, *, timeout: float = 30, python: str = sys.executable):
        self.script = Path(script).resolve(strict=True)
        self.timeout = timeout
        self.python = python

    async def __call__(self, program: Path) -> Evaluation:
        program = Path(program).resolve(strict=True)
        output = program.parent / "evaluation.json"
        output.unlink(missing_ok=True)
        with (
            (program.parent / "stdout.log").open("wb") as stdout,
            (program.parent / "stderr.log").open("wb") as stderr,
        ):
            process = await asyncio.create_subprocess_exec(
                self.python,
                str(self.script),
                "--program",
                str(program),
                "--output",
                str(output),
                cwd=program.parent,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(process.wait(), self.timeout)
            except asyncio.TimeoutError:
                return Evaluation(valid=False, feedback=f"Evaluation timeout after {self.timeout}s")
            finally:
                # Also stop descendants if the evaluator itself already exited.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        if process.returncode:
            raise EvaluationError(
                f"Evaluator exited {process.returncode}; see {program.parent}/stderr.log"
            )
        try:
            return Evaluation.model_validate_json(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError) as exc:
            raise EvaluationError(f"Invalid or missing evaluator result: {output}") from exc
