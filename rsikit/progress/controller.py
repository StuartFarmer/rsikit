"""Run-scoped logging, dashboard coordination, and resource ownership."""

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import RLock

from rich.console import Console
from rich.live import Live

from .model import ProgressState
from .view import DashboardView, _message, _text


class RunDisplay:
    """Consume domain log records; neither runs nor optimizers push UI snapshots."""

    def __init__(self, path, console):
        self.path, self.console = path, console
        self.model = ProgressState()
        self.view = DashboardView(path, console)
        self.lock = RLock()
        self.file = self.live = self.stream_key = None
        self.closed = False

    def open(self):
        self.file = (self.path / "run.log").open("a", encoding="utf-8")

    def consume(self, record):
        stamp = (
            datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        with self.lock:
            if self.closed:
                return
            payload = getattr(record, "progress", None)
            if not isinstance(payload, dict) or payload.get("kind") != "workers":
                self.view.events.append(record, stamp)
            if payload is None:
                return
            try:
                self.model.apply(payload, stamp)
            except (KeyError, TypeError, ValueError) as exc:
                diagnostic = f"Invalid progress record: {exc}"
                self.view.events.append(
                    logging.makeLogRecord(dict(msg=diagnostic, levelno=logging.ERROR)), stamp
                )
                return diagnostic

    def render(self):
        with self.lock:
            return self.view.render(self.model)

    def emit(self, record):
        with self.lock:
            if self.closed:
                return
            stamp = (
                datetime.fromtimestamp(record.created, timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            message = f"{stamp} {record.levelname} [{record.name}] {_message(record)}"
            self.file.write(message + "\n")
            self.file.flush()
            diagnostic = self.consume(record)
            if diagnostic:
                self.file.write(diagnostic + "\n")
                self.file.flush()
                message += "\n" + diagnostic
            if not self.model.active:
                return
        # Live may wait for its refresh thread, which acquires the render lock.
        if self.live is None and self.model.finished is None and self.console.is_terminal:
            stream = self.console.file
            try:
                stream_key = stream.fileno()
            except (AttributeError, OSError):
                stream_key = id(stream)
            if stream_key not in _live_streams:
                self.live = Live(
                    console=self.console, get_renderable=self.render, refresh_per_second=4
                )
                self.stream_key = stream_key
                _live_streams.add(stream_key)
                self.live.start()
        if self.live is None:
            self.console.print(_text(message), soft_wrap=True)
        elif self.model.finished is not None:
            self._stop_live()

    def _stop_live(self):
        live, stream_key = self.live, self.stream_key
        self.live = self.stream_key = None
        try:
            if live is not None:
                live.stop()
        finally:
            _live_streams.discard(stream_key)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.model.close()
            log_file, self.file = self.file, None
        try:
            self._stop_live()
        finally:
            if log_file is not None:
                log_file.close()


_current_run = ContextVar("rsikit_log_run", default=None)
_bindings = []
_logger_settings = {}
_live_streams = set()


class ProgressHandler(logging.Handler):
    def emit(self, record):
        display = _current_run.get()
        if display is not None:
            display.emit(record)


_run_handler = ProgressHandler()


@contextmanager
def bind_run(path, console=None):
    """Bind logging to the Run's task context; lazily acquire a terminal display."""
    display = RunDisplay(path, console or Console())
    display.open()
    if not _bindings:
        for name in ("rsikit", "research"):
            logger = logging.getLogger(name)
            _logger_settings[name] = logger.level
            logger.addHandler(_run_handler)
            logger.setLevel(logging.INFO)
    _bindings.append(display)
    token = _current_run.set(display)
    try:
        yield display
    except BaseException as exc:
        if display.model.active:
            cancelled = isinstance(exc, asyncio.CancelledError)
            logging.getLogger("rsikit").log(
                logging.INFO if cancelled else logging.ERROR,
                "Run cancelled" if cancelled else "Run failed",
                exc_info=not cancelled,
                extra={
                    "progress": dict(
                        kind="search_finished",
                        status="cancelled" if cancelled else "failed",
                        reason="cancelled" if cancelled else str(exc),
                    )
                },
            )
        raise
    finally:
        try:
            display.close()
        finally:
            _current_run.reset(token)
            _bindings.remove(display)
            if not _bindings:
                for name, level in _logger_settings.items():
                    logger = logging.getLogger(name)
                    logger.removeHandler(_run_handler)
                    logger.setLevel(level)
                _logger_settings.clear()
