"""
logging_helpers.py — The coloured logger of the framework.

Taken from https://github.com/mjimcua/logging_helpers (LoggerManager + ColoredFormatter),
with three changes:
  · the handler writes to the CURRENT sys.stdout at every record, so a test that
    redirects the console captures the log
  · colours can be switched off (tests), and the log can also go to a file (no colours)
  · it can configure a NAMED logger instead of the root one, so the framework's log does
    not change the logging of other libraries (pandas, sqlalchemy)

It is called from one place only: Config.__post_init__.

Levels used by the steps:
    DOC      (blue)    documentation: step titles, purposes, explanations, summaries
    INFO     (green)   a check that passed
    WARNING  (yellow)  a check that failed without blocking · a check not evaluated
    ERROR    (red)     a check that failed and blocks the run
DOC is a level of its own (21, between INFO and WARNING): `logger.doc(message)`.
"""

# ─── imports ─────────────────────────────────────────────────────────────────────
import functools
import logging
import sys
from typing import Dict, Optional

from colorama import Back, Fore, Style


# ─── named constants ─────────────────────────────────────────────────────────────
# The documentation level: shown at INFO, hidden at WARNING, blue.
DOC_LEVEL = 21
DOC_LEVEL_NAME = "DOC"
logging.addLevelName(DOC_LEVEL, DOC_LEVEL_NAME)

VALID_LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET", 50, 40, 30, 20, 10, 0]
LOG_FORMAT = "{color}{asctime} | {message}{reset}"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LEVEL_COLORS = {
    "DEBUG": Fore.BLUE,
    "DOC": Fore.BLUE,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
}


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.colors = colors if colors else {}

    def format(self, record) -> str:
        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL if self.colors else ""
        return super().format(record)


class CurrentStdoutHandler(logging.StreamHandler):
    """A StreamHandler that always writes to the sys.stdout of the moment."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        pass    # the stream is always the current sys.stdout


class LoggerManager:

    def __init__(self, log_level, use_colors: bool = True, log_file: Optional[str] = None):
        assert log_level in VALID_LOG_LEVELS, \
            "Incorrect Level. See https://docs.python.org/3/library/logging.html#levels"
        self.log_level = log_level
        self.use_colors = use_colors
        self.log_file = log_file

    def get_logger_configured(self, logger_name: Optional[str] = None) -> logging.Logger:
        console_formatter = ColoredFormatter(LOG_FORMAT, style="{", datefmt=LOG_DATE_FORMAT,
                                             colors=LEVEL_COLORS if self.use_colors else None)
        console_handler = CurrentStdoutHandler()
        console_handler.setFormatter(console_formatter)

        logger = logging.getLogger(logger_name)
        logger.handlers[:] = []
        logger.addHandler(console_handler)
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setFormatter(ColoredFormatter(LOG_FORMAT, style="{", datefmt=LOG_DATE_FORMAT))
            logger.addHandler(file_handler)
        logger.setLevel(self.log_level)
        logger.propagate = logger_name is None     # a named logger does not repeat its lines through the root
        logger.doc = functools.partial(logger.log, DOC_LEVEL)     # logger.doc("...") → blue
        return logger
