"""retry_handler.py — Single Responsibility: Exponential-backoff retry.

Wraps any callable and retries it up to `max_attempts` times using
exponential backoff on failure. Completely agnostic to what it retries.

Retry schedule (default):
  Attempt 1 → fail → wait 1s
  Attempt 2 → fail → wait 2s
  Attempt 3 → fail → wait 4s
  Attempt 4 → raise final exception
"""

import time
from app.utils.logger import logger


class RetryHandler:
    """Execute a callable with exponential-backoff retry on exception."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0):
        """
        Args:
            max_attempts: Total number of tries before giving up.
            base_delay:   Seconds to wait after the first failure.
                          Doubles on every subsequent failure.
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def execute(self, func, *args, **kwargs):
        """
        Call func(*args, **kwargs) with retry logic.

        Returns:
            The return value of func on success.

        Raises:
            The last exception raised by func after all attempts fail.
        """
        last_exc = None
        delay = self.base_delay

        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_attempts:
                    logger.warning(
                        f"[RetryHandler] Attempt {attempt}/{self.max_attempts} "
                        f"failed: {exc}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2          # exponential backoff
                else:
                    logger.error(
                        f"[RetryHandler] All {self.max_attempts} attempts failed. "
                        f"Last error: {exc}"
                    )

        raise last_exc


# Default handler used by the pipeline (3 attempts, 1s base delay)
retry_handler = RetryHandler(max_attempts=3, base_delay=1.0)
