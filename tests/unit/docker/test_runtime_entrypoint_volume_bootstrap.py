# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for runtime fresh-volume bootstrap."""

from __future__ import annotations

import pytest

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]


def test_entrypoint_repairs_runtime_volume_paths_before_the_boot_preflight() -> None:
    """The volume repair and the privilege drop must precede every render.

    OMN-17372 collapsed the schema stamp and both renders into one warm
    interpreter (``omnibase_infra.runtime.entrypoint_preflight``), so the
    ordering guard now anchors on that single launch rather than on the two
    module names it replaced. The invariant is unchanged: nothing writes to
    /app/data before it is owned by the runtime user and privileges are
    dropped.
    """
    entrypoint = (DOCKER_DIR / "entrypoint-runtime.sh").read_text()

    bootstrap_pos = entrypoint.index("Fresh Volume Bootstrap")
    data_dir_pos = entrypoint.index("/app/data/delegation")
    chown_pos = entrypoint.index("chown -R omniinfra:omniinfra")
    drop_pos = entrypoint.index('exec gosu omniinfra "$0" "$@"')
    preflight_pos = entrypoint.index(
        'exec python -m omnibase_infra.runtime.entrypoint_preflight "$@"'
    )

    assert bootstrap_pos < data_dir_pos < chown_pos < drop_pos < preflight_pos
    assert (
        "install -d -o omniinfra -g omniinfra /app/data /app/data/delegation /app/logs /app/tmp"
        in entrypoint
    )
    assert "chown -R omniinfra:omniinfra /app/data /app/logs /app/tmp" in entrypoint


def test_runtime_image_installs_gosu_for_privilege_drop() -> None:
    dockerfile = (DOCKER_DIR / "Dockerfile.runtime").read_text()

    assert "gosu \\" in dockerfile
    assert "USER root" in dockerfile
    assert "dropping to the non-root runtime user" in dockerfile
