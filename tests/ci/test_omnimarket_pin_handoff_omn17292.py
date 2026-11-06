# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay the actual pinned Credentials page through the exposure-reader gate."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import yaml

from omnibase_infra.validators.bus_backed_exposure_readers import (
    collect_local_page_readers,
)

pytestmark = pytest.mark.unit


@pytest.mark.live_contact(
    "tests/ci/fixtures/omnimarket_pin_credentials_reader_omn17292.json"
)
def test_pinned_credentials_page_is_a_reader(
    recorded_response: dict[str, object], tmp_path: Path
) -> None:
    for path, response in recorded_response["responses"].items():
        (tmp_path / Path(path).name).write_bytes(base64.b64decode(response["content"]))
    readers = collect_local_page_readers(tmp_path)
    assert readers["onex.snapshot.projection.tenant-credentials.v1"] == {
        "local-page:credentials.page.yaml"
    }


def test_credentials_recording_is_captured_at_the_omnidash_sibling_pin() -> None:
    """OMN-22552: the replayed page is the one the gate reads at the pinned omnidash."""
    root = Path(__file__).resolve().parents[2]
    pins = yaml.safe_load((root / ".github/sibling-pins.yaml").read_text())
    omnidash_pin = pins["pins"]["omnidash"]
    recording = json.loads(
        (
            root / "tests/ci/fixtures/omnimarket_pin_credentials_reader_omn17292.json"
        ).read_text()
    )
    source = recording["_provenance"]["source"]
    assert f"/omnidash/tree/{omnidash_pin}/" in source
    for path, response in recording["responses"].items():
        assert response["url"].endswith(f"?ref={omnidash_pin}"), path
