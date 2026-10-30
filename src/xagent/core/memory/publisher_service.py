"""System-managed publisher entry point, separate from model execution.

Run python -m xagent.core.memory.publisher_service with the publisher-only
XAGENT_MEMORY_SEAT_PUBLISHER_CONFIG. No end-user key, snapshot UI, organization
write, or automatic account/seat creation is introduced here.
"""

from __future__ import annotations

import json
import logging
import signal
import threading
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine

from ...config import get_memory_seat_publisher_config
from .seat_file_provider import _unique_object
from .seat_publisher import TrustedSeatPublisher
from .seat_verifier import MemsrvSeatVerifier, UserBinding, read_publisher_secret

logger = logging.getLogger(__name__)


def build_publisher(config: dict[str, Any]) -> tuple[TrustedSeatPublisher, Engine]:
    if (
        type(config) is not dict
        or set(config)
        != {
            "version",
            "service_url",
            "control_key_file",
            "runtime_key_file",
            "database_url_file",
            "manifest",
            "journal",
            "bindings",
        }
        or type(config["version"]) is not int
        or config["version"] != 1
    ):
        raise ValueError("invalid system publisher config")
    paths = {}
    for key in (
        "control_key_file",
        "runtime_key_file",
        "database_url_file",
        "manifest",
        "journal",
    ):
        if type(config[key]) is not str or not Path(config[key]).is_absolute():
            raise ValueError("publisher paths must be absolute")
        paths[key] = Path(config[key])
    if type(config["bindings"]) is not list:
        raise ValueError("publisher bindings must be a list")
    bindings = []
    for row in config["bindings"]:
        if type(row) is not dict or set(row) != {"xagent_user_id", "memory_user_id"}:
            raise ValueError("invalid pinned identity binding")
        bindings.append(UserBinding(row["xagent_user_id"], row["memory_user_id"]))
    engine = create_engine(
        read_publisher_secret(paths["database_url_file"]), pool_pre_ping=True
    )
    try:
        verifier = MemsrvSeatVerifier(
            engine=engine,
            service_url=config["service_url"],
            control_key_file=paths["control_key_file"],
            runtime_key_file=paths["runtime_key_file"],
            bindings=tuple(bindings),
        )
        return TrustedSeatPublisher(
            paths["manifest"], paths["journal"], verifier
        ), engine
    except BaseException:
        engine.dispose()
        raise


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    publisher = None
    engine = None
    stop = threading.Event()
    try:
        config = json.loads(
            get_memory_seat_publisher_config().read_bytes(),
            object_pairs_hook=_unique_object,
        )
        publisher, engine = build_publisher(config)
        signal.signal(signal.SIGTERM, lambda *_args: stop.set())
        signal.signal(signal.SIGINT, lambda *_args: stop.set())
        publisher.start()
        logger.info("System memory-seat publisher started")
        while not stop.wait(1):
            if publisher._thread is None or not publisher._thread.is_alive():
                raise RuntimeError("publisher refresh service stopped")
        return 0
    except Exception as error:  # noqa: BLE001 - keep credentials/DB URLs out of operator logs
        logger.error("System memory-seat publisher stopped (%s)", type(error).__name__)
        return 1
    finally:
        if publisher is not None:
            publisher.close()
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
