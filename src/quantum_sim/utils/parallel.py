"""
Parallel mapping over shot files.

Loading ~750 HDF5 shots is I/O and decompression bound, so it parallelises
almost perfectly. The progress bar is optional and auto-detects a notebook, so
the same call works from a script on a machine without ipywidgets installed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = ["run_parallel_tasks"]


def run_parallel_tasks(
    func: Callable[..., Any],
    tasks: Sequence[Iterable[Any]],
    processes: int = 4,
    progress: bool | None = None,
) -> list[Any]:
    """
    Run ``func(*task)`` for each entry of ``tasks`` across worker processes.

    Results are returned in task order. ``progress`` draws an ipywidgets bar
    with an ETA; ``None`` means "on inside a notebook, off elsewhere".
    """
    from multiprocess import Pipe, Process  # noqa: PLC0415 — optional dependency

    tasks = list(tasks)
    if not tasks:
        return []
    processes = max(1, min(processes, len(tasks)))

    reporter = _ProgressReporter(len(tasks), enabled=progress)

    chunk = -(-len(tasks) // processes)  # ceiling division
    chunks = [tasks[i * chunk : (i + 1) * chunk] for i in range(processes)]

    def worker(index, sub_tasks, pipe):
        for i, task in enumerate(sub_tasks):
            pipe.send([index, "DATA", (i, func(*task))])
        pipe.send([index, "RET", None])

    pipe_in, pipe_out = Pipe()
    done_counts = [0] * processes
    results: list[list[Any]] = [[] for _ in range(processes)]
    finished = 0

    procs = [Process(target=worker, args=(i, chunks[i], pipe_in)) for i in range(processes)]
    for proc in procs:
        proc.start()

    try:
        while finished < processes:
            index, kind, payload = pipe_out.recv()
            if kind == "DATA":
                offset, value = payload
                done_counts[index] = offset + 1
                results[index].append(value)
                reporter.update(sum(done_counts))
            elif kind == "RET":
                finished += 1
    finally:
        for proc in procs:
            proc.join()

    reporter.close()
    return [item for chunk_results in results for item in chunk_results]


class _ProgressReporter:
    """ipywidgets progress bar that degrades to silence outside a notebook."""

    def __init__(self, total: int, enabled: bool | None = None):
        self.total = total
        self._bar = None
        self._label = None
        self._start = 0.0
        self._last_report = 0.0

        if enabled is None:
            enabled = _in_notebook()
        if not enabled:
            return

        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            return

        import time

        self._start = time.time()
        self._bar = widgets.IntProgress(value=0, min=0, max=total, description="Loading:")
        self._label = widgets.Label(value="ETA: inf")
        display(self._bar, self._label)

    def update(self, done: int) -> None:
        if self._bar is None:
            return

        import time
        from datetime import timedelta

        now = time.time()
        if now - self._last_report < 0.5:  # do not flood the notebook message bus
            return
        self._last_report = now

        self._bar.value = done
        elapsed = now - self._start
        if done and elapsed > 0:
            eta = timedelta(seconds=int((self.total - done) * elapsed / done))
            self._label.value = f"ETA: {eta}"

    def close(self) -> None:
        if self._bar is None:
            return
        self._bar.value = self.total
        self._bar.bar_style = "success"
        self._label.value = "Done"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
