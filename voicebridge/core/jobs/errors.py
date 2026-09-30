"""Structured pipeline/job errors.

Users see ``stage``, ``code``, ``message`` and ``retryable`` -- never a stack
trace. Unexpected exceptions are mapped to ``INTERNAL_ERROR`` with a generic
message by the job manager and logged server-side with the traceback.
"""

from __future__ import annotations


class PipelineError(Exception):
    def __init__(self, stage: str, code: str, message: str, retryable: bool = False):
        super().__init__(f"{code}: {message}")
        self.stage = stage
        self.code = code
        self.message = message
        self.retryable = retryable

    def to_dict(self) -> dict:
        return {"stage": self.stage, "code": self.code, "message": self.message,
                "retryable": self.retryable}


class JobCancelled(PipelineError):
    def __init__(self, message: str = "Job cancelled."):
        super().__init__("job", "JOB_CANCELLED", message, retryable=False)


class CancelToken:
    """Cooperative cancellation: workers call :meth:`check` between units of work."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def check(self) -> None:
        if self.cancelled:
            raise JobCancelled()
