# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay the pinned dashboard's actual Credentials page through reader discovery."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
import yaml

from omnibase_infra.validators.bus_backed_exposure_readers import (
    collect_local_page_readers,
)

pytestmark = pytest.mark.unit


@pytest.mark.live_contact("tests/ci/fixtures/local_page_readers/credentials.json")
def test_pinned_credentials_binding_is_a_shipped_reader(
    recorded_response: dict[str, object],
    tmp_path: Path,
) -> None:
    provenance = recorded_response["_provenance"]
    assert isinstance(provenance, dict)
    pins = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github/sibling-pins.yaml").read_text()
    )["pins"]
    assert provenance["commit_sha"] == pins["omnidash"]
    response = recorded_response["response"]
    assert isinstance(response, dict)
    for name in ("credentials.page.yaml", "credentials.contracts.yaml"):
        artifact = response[name]
        assert artifact["encoding"] == "base64"
        assert artifact["path"] == f"src/pages/local/{name}"
        (tmp_path / name).write_bytes(base64.b64decode(artifact["content"]))
    assert collect_local_page_readers(tmp_path) == {
        "onex.snapshot.projection.tenant-credentials.v1": {"credentials.page.yaml"},
    }


def test_exposure_workflow_provides_the_local_page_surface() -> None:
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / ".github/workflows/exposure-reader-coverage.yml"
        ).read_text()
    )
    steps = workflow["jobs"]["exposure-reader-coverage"]["steps"]
    checkout = next(
        step
        for step in steps
        if step.get("with", {}).get("repository") == "OmniNode-ai/omnidash"
    )
    assert "src/pages/local" in checkout["with"]["sparse-checkout"].splitlines()
    assert any(
        "--local-pages-dir omnidash/src/pages/local" in step.get("run", "")
        for step in steps
    )
