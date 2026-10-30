"""Host lifecycle wiring, with synthetic authorization publications only."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi import FastAPI

from xagent.core.memory import seat_file_provider
from xagent.web import app as app_module
from xagent.web.startup_admission import register_host_startup_admission


@pytest.mark.asyncio
async def test_web_installs_after_host_admission_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.web.test_startup_admission import _patch_runtime_starts

    app = FastAPI()
    events: list[str] = []
    monkeypatch.setattr(app_module, "init_db", lambda: events.append("database"))
    _patch_runtime_starts(monkeypatch, events)
    monkeypatch.setattr(
        seat_file_provider, "start_memory_seat_runtime", lambda: events.append("seats")
    )

    async def host_admission() -> None:
        events.append("trusted host")

    register_host_startup_admission(app, host_admission)
    await app_module._initialize_database_and_admit_runtime(app)
    assert (
        events.index("trusted host") < events.index("seats") < events.index("runtime")
    )
    assert events.index("seats") < events.index("task admission")


@pytest.mark.parametrize("signal_name", ["worker_process_init", "worker_ready"])
def test_celery_installs_in_execution_processes(
    monkeypatch: pytest.MonkeyPatch, signal_name: str
) -> None:
    from celery import signals

    from xagent.web.jobs import celery_app  # noqa: F401 - connects signal handlers

    start = Mock()
    monkeypatch.setattr(seat_file_provider, "start_memory_seat_runtime", start)
    getattr(signals, signal_name).send(sender=Mock(pool=object()))
    start.assert_called_once_with()


def test_prefork_parent_does_not_start_refresh_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from celery.concurrency.prefork import TaskPool
    from celery.signals import worker_ready

    from xagent.web.jobs import celery_app  # noqa: F401 - connects signal handlers

    start = Mock()
    monkeypatch.setattr(seat_file_provider, "start_memory_seat_runtime", start)
    worker_ready.send(sender=Mock(pool=TaskPool(1)))
    start.assert_not_called()


@pytest.mark.parametrize("signal_name", ["worker_process_shutdown", "worker_shutdown"])
def test_celery_revokes_when_process_stops(
    monkeypatch: pytest.MonkeyPatch, signal_name: str
) -> None:
    from celery import signals

    from xagent.web.jobs import celery_app  # noqa: F401 - connects signal handlers

    stop = Mock()
    monkeypatch.setattr(seat_file_provider, "stop_memory_seat_runtime", stop)
    getattr(signals, signal_name).send(sender=None)
    stop.assert_called_once_with()
