# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability runtime mounts the lane declaration its gateway door resolves a broker from (OMN-26121).

The lab overlay for ``node_gateway_batch_ingest_effect`` names ``/etc/onex/workspace-config`` as the
mounted config root, and the door's lifespan reads ``config/lab/ci_bus_lanes.yaml`` under it. The dev
runtime mounts that folder (OMN-24854); the stability runtime did not, so after the stability follow
rolled omnimarket's door in, the runtime answered 503 on /health with ``DelegateLaneSelectionError:
the lane declaration /etc/onex/workspace-config/config/lab/ci_bus_lanes.yaml does not exist`` and the
agent rolled the lane back. These checks cover the shipped compose entry and the refresh script that
exports the folder; the running container is read on the lab after the stability follow.
"""

from __future__ import annotations

import pytest

from tests.ci._compose_render_omn21099 import REPO_ROOT, load_compose

pytestmark = [pytest.mark.unit]

STABILITY_LANE = "-".join(("stability", "test"))
STABILITY_COMPOSE = f"docker-compose.{STABILITY_LANE}.yml"
RUNTIME_SERVICE = "omninode-runtime"
MOUNT_TARGET = "/etc/onex/workspace-config/config/lab"
REQUIRED_SOURCE_PREFIX = "${ONEX_LANE_DECLARATION_DIR:?"
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"


def _runtime_volumes() -> list[str]:
    service = load_compose(STABILITY_COMPOSE)["services"][RUNTIME_SERVICE]
    volumes = service["volumes"]
    assert isinstance(volumes, list)
    return [str(volume) for volume in volumes]


def test_stability_runtime_mounts_the_lane_declaration_read_only() -> None:
    mounts = [v for v in _runtime_volumes() if MOUNT_TARGET in v]
    assert len(mounts) == 1, mounts
    assert mounts[0].startswith(REQUIRED_SOURCE_PREFIX), mounts[0]
    assert mounts[0].endswith(f":{MOUNT_TARGET}:ro"), mounts[0]


def test_control_the_mount_is_found_where_the_dev_lane_carries_it() -> None:
    """Positive control: the same read finds the dev lane's mount, so a zero above is a real absence."""
    dev = load_compose("docker-compose.dev-lane.yml")["services"]
    carriers = [
        name
        for name, service in dev.items()
        if isinstance(service, dict)
        and any(MOUNT_TARGET in str(v) for v in service.get("volumes", []))
    ]
    assert carriers, "the dev lane carries the lane declaration mount"


def test_stability_refresh_exports_the_declaration_folder() -> None:
    text = REFRESH_SCRIPT.read_text(encoding="utf-8")
    assert '"${STATE_DIR}" with-declaration)' in text
    assert "without-declaration" not in text
