"""Errors raised by volightcurve file I/O (Ticket 8).

The application bridge maps these to ``PipeException`` for UI surfaces.
"""

from __future__ import annotations


class LightcurveIOError(ValueError):
    """Raised when lightcurve file parse or serialise fails.

    Args:
        message (str): Human-readable reason (British English for user-facing text).
    """

    def __init__(self, message: str):
        """Initialises the error with a clear message.

        Args:
            message (str): Failure reason.
        """
        super().__init__(message)
