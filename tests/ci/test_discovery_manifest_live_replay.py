# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay the deployed manifest through the additive discovery schema (OMN-18713)."""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest


@pytest.mark.live_contact("tests/ci/fixtures/discovery_manifest_live.json")
def test_deployed_manifest_without_skip_rows_remains_readable(
    recorded_response: dict[str, object],
) -> None:
    """Old runtime responses remain readable without requiring new skip fields."""
    manifest = ModelAutoWiringManifest.model_validate(recorded_response["response"])
    assert manifest.total_discovered == 1
    assert manifest.total_errors == 1
    assert manifest.total_skips == 0
    assert manifest.contracts[0].contract_content_hash
    assert manifest.errors[0].reason == "discovery_error"
    assert manifest.errors[0].contract_path is None
    reloaded = ModelAutoWiringManifest.model_validate_json(manifest.model_dump_json())
    assert reloaded == manifest
