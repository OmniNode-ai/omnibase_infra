# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the runtime build-source contract in compose."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
def test_runtime_services_use_build_source_arg() -> None:
    compose = yaml.safe_load(
        Path("docker/docker-compose.infra.yml").read_text(encoding="utf-8")
    )
    runtime_service = compose["x-runtime-base"]
    build_args = runtime_service["build"]["args"]

    assert build_args["BUILD_SOURCE"] == "${BUILD_SOURCE:-release}"
    assert (
        build_args["RELEASE_MANIFEST_PATH"]
        == "${RELEASE_MANIFEST_PATH:-docker/runtime-release-manifest.json}"
    )
