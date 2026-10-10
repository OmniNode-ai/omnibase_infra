# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay the actual pinned Credentials page through the exposure-reader gate."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

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
