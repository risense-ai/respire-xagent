"""Test package setup must normalize proxies before client imports."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("localhost,[::1], 127.0.0.1,", "localhost,::1,127.0.0.1"),
        ("localhost,127.0.0.1", "localhost,127.0.0.1"),
        ("", ""),
    ],
)
def test_package_bootstrap_precedes_http_client_import(
    raw: str, normalized: str
) -> None:
    env = dict(os.environ, NO_PROXY=raw, no_proxy=raw)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tests, os; "
            f"assert os.environ['NO_PROXY'] == {normalized!r}; "
            f"assert os.environ['no_proxy'] == {normalized!r}; "
            "import httpx; client = httpx.Client(); client.close()",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
