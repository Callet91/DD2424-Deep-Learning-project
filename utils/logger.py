"""Logger function."""

import logging


def set_logger():
    """Initialize logger and set logger file."""
    logging.basicConfig(filename="logs.log", level=logging.DEBUG)
    log = logging.getLogger("logs.log")
    return log


_LOGGER = set_logger()
