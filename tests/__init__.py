"""Initialize the test environment before conftest imports HTTP clients."""

import os


def _sanitize_no_proxy_env() -> None:
    """Normalize bracketed IPv6 before httpx builds environment mount patterns.

    Pytest imports this package before tests.conftest. Keep this bootstrap here
    so the conftest imports stay at module level without changing their order.
    """
    for name in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(name)
        if not raw or "[" not in raw:
            continue
        os.environ[name] = ",".join(
            part.replace("[", "").replace("]", "")
            for part in (p.strip() for p in raw.split(","))
            if part
        )


_sanitize_no_proxy_env()
