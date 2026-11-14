# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23981: the stamp handler and the publish seam agree on real envelopes."""

from __future__ import annotations

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.handlers.handler_event_provenance_stamp.handler import (
    HandlerEventProvenanceStamp,
    read_provenance_overlay,
)
from omnibase_infra.models.event_bus.model_event_provenance_stamp_request import (
    ModelEventProvenanceStampRequest,
)
from omnibase_infra.models.event_bus.util_event_provenance_seam import (
    dump_envelope_json,
)

pytestmark = pytest.mark.integration

_ENV = {
    "ONEX_PROVENANCE_LANE": "dev",
    "ONEX_PROVENANCE_HOST": "lab-host-201",
    "ONEX_PROVENANCE_INSTALL_SOURCE": "branch",
    "ONEX_PROVENANCE_REVISION": "a09066caf945cd79e13447d92ad42549f1019f78",
}


def test_handler_and_seam_stamp_the_same_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)
    envelope = ModelEventEnvelope[dict[str, str]](payload={"k": "v"})
    result = HandlerEventProvenanceStamp().handle(
        ModelEventProvenanceStampRequest(
            envelope=envelope, overlay=read_provenance_overlay()
        )
    )
    assert result.stamped is True
    wire = dump_envelope_json(envelope)
    assert isinstance(wire, dict)
    assert wire["provenance"] == result.envelope.model_dump(mode="json")["provenance"]
