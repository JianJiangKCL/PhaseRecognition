from functools import wraps
import time
from typing import Callable, Any

class RateLimiter:
    def __init__(self, calls: int, period: int, delay: int = 0):
        """
        Initialize rate limiter.
        
        Args:
            calls (int): Number of calls allowed in the period
            period (int): Time period in seconds
            delay (int): Minimum delay between calls in seconds
        """
        self.calls = calls
        self.period = period
        self.delay = delay
        self.timestamps = []

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Add minimum delay between calls
            if self.delay > 0:
                time.sleep(self.delay)

            # Remove timestamps outside the current period
            now = time.time()
            self.timestamps = [ts for ts in self.timestamps if ts > now - self.period]

            # If we've hit the rate limit, sleep until oldest timestamp expires
            if len(self.timestamps) >= self.calls:
                sleep_time = self.timestamps[0] - (now - self.period)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.timestamps = self.timestamps[1:]

            # Add current timestamp and execute function
            self.timestamps.append(time.time())
            return func(*args, **kwargs)

        return wrapper

# Predefined rate limiters
def openai_rate_limit(func: Callable) -> Callable:
    return RateLimiter(calls=8, period=60, delay=7)(func)

def xai_rate_limit(func: Callable) -> Callable:
    return RateLimiter(calls=45, period=3600, delay=3)(func)

def google_rate_limit(func: Callable) -> Callable:
    return RateLimiter(calls=8, period=60, delay=7)(func)

def anthropic_rate_limit(func: Callable) -> Callable:
    return RateLimiter(calls=8, period=60, delay=7)(func) 