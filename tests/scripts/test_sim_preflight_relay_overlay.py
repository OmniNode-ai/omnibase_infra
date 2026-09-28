# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The optional relay transport cannot acquire credentials or host access."""

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from scripts.runtime_build.verify_sim_preflight_isolation import verify
from tests.scripts.test_sim_preflight_isolation_omn19726 import _config


def _relay() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[2]
        / "docker/docker-compose.sim-preflight-relay.yml"
    )
    return cast(
        "dict[str, Any]",
        yaml.safe_load(path.read_text())["services"]["relay-transport"],
    )


def test_relay_transport_is_optional_and_restricted() -> None:
    config = _config()
    config["services"]["relay-transport"] = _relay()
    assert verify(config)["service_count"] == 14


def test_relay_health_checks_inert_process_instead_of_inherited_http() -> None:
    relay = _relay()
    health = relay["healthcheck"]
    assert health["test"] == [
        "CMD",
        "python",
        "-c",
        "import os; assert os.getuid() == 1000; assert b'import signal; signal.pause()' in open('/proc/1/cmdline', 'rb').read().split(bytes([0]))",
    ]
    assert health["interval"] == "30s"
    assert health["timeout"] == "3s"
    assert health["start_period"] == "5s"
    assert health["retries"] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment", {"TOKEN": "not-a-secret"}),
        ("secrets", [{"source": "forbidden-secret"}]),
        ("configs", [{"source": "forbidden-config"}]),
        (
            "volumes",
            [
                {
                    "type": "bind",
                    "source": "/forbidden-input",
                    "target": "/input",
                    "read_only": True,
                }
            ],
        ),
        ("ports", [{"host_ip": "127.0.0.1", "published": "9999"}]),
        ("read_only", False),
        ("cap_drop", []),
        ("security_opt", []),
        ("user", "0"),
        ("entrypoint", ["sh"]),
        ("command", ["arbitrary"]),
        ("healthcheck", {"test": ["CMD", "python", "-c", "print('arbitrary')"]}),
        ("healthcheck", {"disable": True}),
    ],
)
def test_relay_transport_refuses_extra_authority(field: str, value: object) -> None:
    config = _config()
    relay = _relay()
    relay[field] = value
    config["services"]["relay-transport"] = relay
    with pytest.raises(ValueError):
        verify(config)
