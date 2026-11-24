# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the four Infisical names on the verify-class runners (OMN-19812).

The scheduled C28 consumer-flow probe runs on the `omnibase-verify` class and
reads the stability lane's Kafka SASL secret from Infisical through a machine
identity. The runner holds that identity only if its container environment
carries four names: INFISICAL_ADDR, INFISICAL_PROJECT_ID, INFISICAL_CLIENT_ID
and INFISICAL_CLIENT_SECRET. Without them the probe cannot look and grades
INDETERMINATE.

Only the names live in the compose file. Each value is interpolated from the
deployed `.env` beside it, in the default form `${NAME:-}`: the runners compose
file is interpolated whole, so a `:?` guard on a new name would break every
other service when the name is unset, while `${NAME:-}` leaves an unset name
empty.

The three members are interchangeable by construction, so the names are pinned
on all three and the members are compared field by field after the three
identity fields are normalised away.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.runners.yml"

VERIFY_SERVICES = (
    "omninode-verify-runner-1",
    "omninode-verify-runner-2",
    "omninode-verify-runner-3",
)
INFISICAL_NAMES = (
    "INFISICAL_ADDR",
    "INFISICAL_PROJECT_ID",
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
)


def _compose() -> dict[str, Any]:
    loaded = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _service(name: str) -> dict[str, Any]:
    services = _compose()["services"]
    assert name in services, f"{name} is not defined in {COMPOSE_FILE}"
    service = services[name]
    assert isinstance(service, dict)
    return service


def _environment(name: str) -> dict[str, str]:
    env = _service(name).get("environment", {})
    assert isinstance(env, dict), f"{name} uses list-form environment; update this test"
    return {str(k): str(v) for k, v in env.items()}


def _names_not_passed(env: dict[str, str]) -> list[str]:
    """The Infisical names an environment does not pass in the default form."""
    return [n for n in INFISICAL_NAMES if env.get(n) != f"${{{n}:-}}"]


def _normalise(service_name: str, service: dict[str, Any]) -> str:
    """The service with its identity fields replaced by a fixed placeholder.

    The identity fields are the container name, `RUNNER_NAME` and the creds
    volume; each embeds the member's own service name, so one substitution
    covers all three.
    """
    dumped = json.dumps(service, sort_keys=True)
    return dumped.replace(service_name, "omninode-verify-runner-N").replace(
        service_name.removeprefix("omninode-"), "verify-runner-N"
    )


def _normalised(service_name: str) -> str:
    return _normalise(service_name, _service(service_name))


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
@pytest.mark.parametrize("name", INFISICAL_NAMES)
def test_verify_runner_passes_each_infisical_name_in_the_default_form(
    verify_service: str, name: str
) -> None:
    env = _environment(verify_service)
    assert name in env, f"{verify_service} does not pass {name}"
    assert env[name] == f"${{{name}:-}}", (
        f"{verify_service} passes {name} as {env[name]!r}, not the default form"
    )


@pytest.mark.parametrize("verify_service", VERIFY_SERVICES)
def test_verify_runner_holds_no_literal_infisical_value(verify_service: str) -> None:
    """Values never enter the repo: every Infisical entry is an interpolation."""
    for key, value in _environment(verify_service).items():
        if key.startswith("INFISICAL_"):
            assert value.startswith("${") and value.endswith(":-}"), (
                f"{verify_service} {key} is not an interpolation"
            )


def test_verify_runners_differ_only_in_their_identity_fields() -> None:
    first = VERIFY_SERVICES[0]
    for name in VERIFY_SERVICES[1:]:
        assert _normalised(first) == _normalised(name), (
            f"{name} diverges from {first} beyond its identity fields"
        )


def test_the_check_bites_when_a_name_is_removed() -> None:
    """Positive control: the predicate the tests above rest on can fail.

    Without this, a parse that returned an empty environment or a predicate that
    accepted anything would report every service as satisfied.
    """
    env = _environment(VERIFY_SERVICES[0])
    assert _names_not_passed(env) == []
    for name in INFISICAL_NAMES:
        without = {k: v for k, v in env.items() if k != name}
        assert _names_not_passed(without) == [name]
    literal = {**env, "INFISICAL_ADDR": "http://example.invalid:8880"}
    assert _names_not_passed(literal) == ["INFISICAL_ADDR"]


def test_the_normalisation_bites_when_one_member_diverges() -> None:
    """Positive control for the identity comparison.

    Dropping one environment entry from one member must change its normalised form, so the
    comparison above cannot pass by normalising everything away.
    """
    name = VERIFY_SERVICES[2]
    service = _service(name)
    trimmed = {
        **service,
        "environment": {
            k: v for k, v in service["environment"].items() if k != "LANE_PROBE_HOST"
        },
    }
    assert _normalise(name, trimmed) != _normalised(name)
    assert _normalised(VERIFY_SERVICES[0]) == _normalised(VERIFY_SERVICES[1])
