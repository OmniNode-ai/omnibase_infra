# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for runtime fresh-volume bootstrap."""

from __future__ import annotations

import pytest

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]


def test_entrypoint_repairs_runtime_volume_paths_before_bifrost_render() -> None:
    entrypoint = (DOCKER_DIR / "entrypoint-runtime.sh").read_text()

    bootstrap_pos = entrypoint.index("Fresh Volume Bootstrap")
    data_dir_pos = entrypoint.index("/app/data/delegation")
    chown_pos = entrypoint.index("chown -R omniinfra:omniinfra")
    drop_pos = entrypoint.index('exec gosu omniinfra "$0" "$@"')
    render_pos = entrypoint.index("render_bifrost_delegation_contract")
    secret_render_pos = entrypoint.index("render_secret_resolver_config")

    assert bootstrap_pos < data_dir_pos < chown_pos < drop_pos < render_pos
    assert drop_pos < secret_render_pos
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
