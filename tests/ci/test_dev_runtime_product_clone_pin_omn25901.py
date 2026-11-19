# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev runtime declares the pin file its product clones are measured against (OMN-25901).

THE RECORDED REFUSAL
--------------------
A dry run of ``dod_verify`` for OMN-25709 inside the dev runtime ends ``skipped``:
five of nine checks cannot be evaluated, ``PRODUCT_CLONE_NOT_FRESH``, because the
verifier fetches ``origin dev`` into a clone mounted read-only and
``DOD_VERIFY_PRODUCT_CLONE_PIN_FILE`` is set nowhere in the runtime
(``fixtures/dev_runtime_product_clone_pin_refusal_omn25901.json``).

THE DECLARATION
---------------
Both compose files that mount the product clones (the Done guard's host service in
each) set that variable to a file in a directory they mount read-only, relative to
``docker/`` like the clones. The deploy agent writes the file from each clone's own
HEAD when a dev job starts (``scripts/deploy-agent/tests/unit/
test_product_clone_pin_writer_omn25901.py``); no revision is typed into a compose file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = [pytest.mark.unit]

DOCKER = Path(__file__).resolve().parents[2] / "docker"
FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "dev_runtime_product_clone_pin_refusal_omn25901.json"
)
PIN_ENV = "DOD_VERIFY_PRODUCT_CLONE_PIN_FILE"
PIN_MOUNT_TARGET = "/etc/onex/product-clone-pin"
PIN_MOUNT = f"../../.onex_state/product-clone-pin:{PIN_MOUNT_TARGET}:ro"

HOSTS: tuple[tuple[str, str], ...] = (
    ("docker-compose.dev-lane.yml", "runtime-effects"),
    ("docker-compose.dev-tenant-fold.yml", "omninode-runtime"),
)


def _fixture() -> dict[str, object]:
    loaded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _service(filename: str, service: str) -> dict[str, object]:
    found = load_compose(DOCKER / filename)["services"][service]
    assert isinstance(found, dict)
    return found


def test_recorded_refusal_names_the_unset_pin_file_variable() -> None:
    """Positive control: the fixture is the refusal this change answers."""
    fixture = _fixture()
    runtime = fixture["runtime"]
    dry_run = fixture["dry_run"]
    assert isinstance(runtime, dict) and isinstance(dry_run, dict)
    assert runtime["pin_file_variable"] == PIN_ENV
    assert runtime["pin_file_variable_in_container_environment"] is False
    assert dry_run["check_message_code"] == "PRODUCT_CLONE_NOT_FRESH"
    assert dry_run["status"] == "skipped"
    assert "Read-only file system" in str(dry_run["check_message_cause"])


@pytest.mark.parametrize(("filename", "service"), HOSTS, ids=[h[1] for h in HOSTS])
def test_pin_file_variable_is_set_in_a_directory_the_service_mounts(
    filename: str, service: str
) -> None:
    """The variable names a file under the mounted pin directory, never a bare value."""
    environment = _service(filename, service)["environment"]
    assert isinstance(environment, dict)
    assert environment.get(PIN_ENV) == f"{PIN_MOUNT_TARGET}/pins.json"


@pytest.mark.parametrize(("filename", "service"), HOSTS, ids=[h[1] for h in HOSTS])
def test_pin_directory_is_mounted_read_only_beside_the_clones(
    filename: str, service: str
) -> None:
    volumes = [str(v) for v in _service(filename, service)["volumes"]]
    assert volumes.count(PIN_MOUNT) == 1
    clones = [v for v in volumes if v.endswith(":ro") and ":/app/omni" in v]
    assert len(clones) == len(_fixture()["mounted_clones"])


@pytest.mark.parametrize(("filename", "service"), HOSTS, ids=[h[1] for h in HOSTS])
def test_no_revision_is_typed_into_the_compose_file(
    filename: str, service: str
) -> None:
    """No 40-hex revision sits in the pin declaration or anywhere in the service."""
    text = json.dumps(_service(filename, service))
    assert not re.search(r"\b[0-9a-f]{40}\b", text)
