# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay actual dashboard bindings through the changed reader collector."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.validators.bus_backed_exposure_readers import (
    ReaderSurfaceError,
    collect_local_page_readers,
)

pytestmark = pytest.mark.unit


@pytest.mark.live_contact(
    "tests/ci/fixtures/omnimarket_pin_credentials_reader_omn17292.json"
)
def test_recorded_credentials_binding_counts_only_its_declared_topic(
    recorded_response: dict[str, Any], tmp_path: Path
) -> None:
    for name, response in recorded_response["responses"].items():
        content = base64.b64decode(response["content"])
        blob = b"blob " + str(len(content)).encode() + b"\0" + content
        assert hashlib.sha1(blob).hexdigest() == response["sha"]
        (tmp_path / Path(name).name).write_bytes(content)
    topic = "onex.snapshot.projection.tenant-credentials.v1"
    assert collect_local_page_readers(tmp_path)[topic] == {
        "local-page:credentials/credentials-keys"
    }

    contract = tmp_path / "credentials.contracts.yaml"
    document = json.loads(contract.read_text())
    document["components"][0]["data_bindings"] = []
    contract.write_text(json.dumps(document))
    assert topic not in collect_local_page_readers(tmp_path)

    contract.write_text("components: [unclosed")
    with pytest.raises(ReaderSurfaceError, match="cannot read the local page contract"):
        collect_local_page_readers(tmp_path)
