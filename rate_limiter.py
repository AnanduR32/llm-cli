"""
Simple rate limiter using a sliding window algorithm.

Tracks request timestamps per key (e.g. per client IP or API key) and
rejects requests that exceed the configured limit within the window.

Usage:
    limiter = RateLimiter(max_requests=10, window_seconds=60)

    if limiter.allow("client-1"):
        # proceed with request
    else:
        # return 429 or wait
"""

import time
import threading
from collections import defaultdict


class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Args:
            max_requests: Maximum number of requests allowed per window.
            window_seconds: Size of the sliding window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def _cleanup(self, key: str) -> None:
        """Remove timestamps outside the current window."""
        cutoff = time.time() - self.window_seconds
        self._requests[key] = [
            ts for ts in self._requests[key] if ts > cutoff
        ]

    def allow(self, key: str = "default") -> bool:
        """
        Check if a request is allowed for the given key.

        Returns True if the request is allowed, False if rate limited.
        This method also records the request timestamp if allowed.
        """
        with self._lock:
            self._cleanup(key)
            if len(self._requests[key]) < self.max_requests:
                self._requests[key].append(time.time())
                return True
            return False

    def wait_time(self, key: str = "default") -> float:
        """
        Returns how many seconds the caller should wait before retrying.
        Returns 0 if the request is allowed now.
        """
        with self._lock:
            self._cleanup(key)
            if len(self._requests[key]) < self.max_requests:
                return 0.0
            # The oldest request in the window determines when a slot opens
            oldest = self._requests[key][0]
            return max(0.0, (oldest + self.window_seconds) - time.time())

    def remaining(self, key: str = "default") -> int:
        """Number of requests remaining in the current window."""
        with self._lock:
            self._cleanup(key)
            return max(0, self.max_requests - len(self._requests[key]))


# Module-level singleton for easy import
_default_limiter: RateLimiter | None = None


def get_limiter(max_requests: int = 10, window_seconds: int = 60) -> RateLimiter:
    """Get or create the module-level rate limiter singleton."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
    return _default_limiter
