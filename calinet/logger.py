# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys
import joblib
import logging
import tqdm as tqmod
from pathlib import Path

import contextvars
from contextlib import contextmanager

from typing import Optional, Union

current_subject = contextvars.ContextVar("current_subject", default="-")

class AnsiColorFormatter(logging.Formatter):
    # ANSI escape codes
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow/orange-ish
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[1;31m" # bold red
    }

    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        msg = super().format(record)
        if not self.use_color:
            return msg

        color = self.COLORS.get(record.levelno)
        if not color:
            return msg
        return f"{color}{msg}{self.RESET}"


class SubjectFilter(logging.Filter):
    def filter(self, record):
        record.subject = current_subject.get("-")
        return True
    

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


def init_logging(
        level: int=logging.INFO,
        logfile: Optional[Union[str, Path]]=None,
        use_tqdm: bool=True,
        use_color: bool=True,
        filemode: str="w"
    ) -> logging.Logger:
    """
    Initialize root logging handlers and formatting.

    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level applied to the root logger and all configured handlers.
    logfile : str, pathlib.Path, or None, optional
        Path to an optional log file. If provided, a file handler is added in
        addition to the stream or tqdm-safe handler.
    use_tqdm : bool, default=True
        Whether to use ``TqdmSafeHandler`` for console logging. If ``False``,
        ``logging.StreamHandler`` writing to ``sys.stderr`` is used instead.
    use_color : bool, default=True
        Whether ANSI color formatting is enabled for the console handler via
        ``AnsiColorFormatter``.
    filemode : str, default="w"
        File open mode used when creating ``logging.FileHandler`` for
        ``logfile``.

    Returns
    -------
    root : logging.Logger
        Configured root logger.

    Notes
    -----
    This function clears all existing handlers from the root logger before
    attaching new handlers.

    A shared ``SubjectFilter`` is added to each configured handler.

    Console logging uses the format
    ``"[%(asctime)s.%(msecs)03d] [%(subject)s] [%(levelname)s] %(name)s - %(message)s"``
    with date format ``"%Y-%m-%d %H:%M:%S"``.

    If ``logfile`` is provided, this function opens or truncates the target
    file according to ``filemode`` and writes log output to it using UTF-8
    encoding.

    This function modifies global logging configuration, captures Python
    warnings via ``logging.captureWarnings(True)``, and may create or overwrite
    a log file on disk.
    """

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    subject_filter = SubjectFilter()

    handler = TqdmSafeHandler() if use_tqdm else logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.addFilter(subject_filter)

    base_fmt = "[%(asctime)s.%(msecs)03d] [%(subject)s] [%(levelname)s] %(name)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(AnsiColorFormatter(base_fmt, datefmt=date_fmt, use_color=use_color))
    root.addHandler(handler)

    if logfile:
        fh = logging.FileHandler(logfile, mode=filemode, encoding="utf-8")
        fh.setLevel(level)
        fh.addFilter(subject_filter)
        fh.setFormatter(logging.Formatter(base_fmt, datefmt=date_fmt))
        root.addHandler(fh)

    logging.captureWarnings(True)
    return root


def worker_init(
        log_dir: Union[str, Path],
        log_level: int
    ) -> None:
    """
    Initialize per-worker logging configuration.

    Parameters
    ----------
    log_dir : str or pathlib.Path
        Directory where worker-specific log files are written.
    log_level : int
        Logging level applied to the worker logger configuration.

    Returns
    -------
    None

    Notes
    -----
    A log file named ``"log.worker.<pid>.log"`` is created or appended to in
    ``log_dir``, where ``<pid>`` is the current process ID.

    This function calls ``init_logging`` with ``use_tqdm=False`` and
    ``filemode="a"`` to ensure logs are appended.

    A logger named ``"calinet.worker"`` is used to emit a startup message
    indicating the worker PID and log file path.

    This function performs file I/O, modifies global logging configuration,
    and writes log output.
    """

    pid = os.getpid()
    worker_log = os.path.join(log_dir, f"log.worker.{pid}.log")

    init_logging(
        level=log_level,
        logfile=worker_log,
        use_tqdm=False,
        use_color=True,
        filemode="a"
    )

    logger = logging.getLogger("calinet.worker")
    logger.info(f"Worker {pid} logging to {worker_log}")


class TqdmSafeHandler(
        logging.Handler
    ):
    """
    Logging handler that writes messages without breaking tqdm progress bars.

    Notes
    -----
    The ``emit`` method formats the log record and attempts to write using
    ``tqmod.write`` if available, preserving tqdm progress bar rendering.

    If ``tqmod.write`` is not available or not callable, the message is written
    directly to ``sys.stderr`` with a newline and flushed immediately.

    This handler performs console I/O and does not manage files or external
    resources.
    """
    
    def emit(self, record):
        msg = self.format(record)
        write = getattr(tqmod, "write", None)
        if callable(write):
            write(msg)               # keeps progress bars intact
        else:
            sys.stderr.write(msg + "\n")  # fallback if write is missing
            sys.stderr.flush()