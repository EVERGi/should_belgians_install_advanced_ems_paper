"""Small progress-bar helpers shared by the command line tool.

Bars are opt-in: library code and the HPC scripts stay silent unless
``set_progress_enabled(True)`` has been called (``belgian_dwellings.cmd_exec`` does
this on startup). Even when enabled, tqdm is told to disable itself when the output
is not a terminal, so redirecting to a log file keeps the log clean.
"""

import sys

from tqdm.auto import tqdm

_ENABLED = False
_WORKER = False

# Keep every bar on one tidy line: "<desc> |####----| 42% 00:12<00:30".
# The bar is given a fixed width so it does not jitter as the elapsed/remaining
# strings grow and shrink.
_BAR_FORMAT = "{desc} |{bar:16}| {percentage:3.0f}% {elapsed}<{remaining}"


def set_progress_enabled(enabled):
    """Turn progress bars on or off process-wide."""
    global _ENABLED
    _ENABLED = bool(enabled)


def progress_enabled():
    return _ENABLED


def set_worker_mode(worker):
    """Mark this process as a pool worker running underneath a parent's bar.

    Set in the parent just before forking so the children inherit it. Worker output
    cannot go through ``tqdm.write``, since the bar object lives in another process;
    instead ``log`` erases the bar's line before writing so the two do not overlap.
    """
    global _WORKER
    _WORKER = bool(worker)


def progress_bar(total, desc, leave=False):
    """Return a tqdm bar, or a no-op stand-in when progress is disabled.

    ``disable=None`` makes tqdm silence itself when stderr is not a terminal.
    """
    return tqdm(
        total=total,
        desc=desc,
        leave=leave,
        disable=None if _ENABLED else True,
        bar_format=_BAR_FORMAT,
        file=sys.stderr,
        ncols=80,
    )


def simulation_bar(microgrid, end_time, desc):
    """Progress bar over the timesteps of a microgrid simulation.

    The total is derived from the simulation horizon, so the bar advances once per
    ``simulation_step()``; call ``bar.update(1)`` inside the loop.
    """
    remaining = end_time - microgrid.utc_datetime
    total = max(int(remaining / microgrid.time_step), 1)
    return progress_bar(total, desc)


def log(message):
    """Print a message without corrupting an in-flight progress bar."""
    if _ENABLED:
        tqdm.write(message)
    elif _WORKER and sys.stderr.isatty():
        # "\r\033[K" = go to column 0 and erase the line the parent's bar sits on.
        # The bar redraws itself on its next update.
        sys.stderr.write("\r\033[K" + message + "\n")
        sys.stderr.flush()
    else:
        print(message)
