# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolution-equivalence tests for the OTEL tracing endpoint descriptor.

OMN-13558 Wave-1 endpoint→overlay migration. Proves the overlay-resolved
``descriptor.otel_exporter_otlp_endpoint`` returns exactly the value the old
direct ``os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")`` read returned for
the same env, across dev / stability / prod lane values, and that an unset var
resolves to the empty string (preserving the opt-in "tracing disabled"
semantics of OMN-3811 — NOT a fail-closed raise).
"""

from __future__ import annotations

import os

import pytest

from omnibase_infra.runtime.tracing_contract_descriptor import (
    contract_otel_exporter_endpoint,
)

_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"

# Per-lane values that the old `os.environ` read would have returned.
_LANE_ENDPOINTS = [
    pytest.param("http://phoenix:6006", id="dev"),
    pytest.param("http://omninode-infra-stability-test-phoenix:6006", id="stability"),
    pytest.param("http://omninode-infra-prod-phoenix:6006", id="prod"),
]


@pytest.fixture(autouse=True)
def _clean_otel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts from an unset endpoint var."""
    monkeypatch.delenv(_ENV, raising=False)


@pytest.mark.unit
@pytest.mark.parametrize("endpoint", _LANE_ENDPOINTS)
def test_descriptor_resolves_same_value_as_direct_env_read(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlay-resolved endpoint == the old direct os.environ read, per lane."""
    monkeypatch.setenv(_ENV, endpoint)

    # Old behavior reproduced inline for equivalence.
    legacy = os.environ.get(_ENV, "").strip()
    resolved = contract_otel_exporter_endpoint()

    assert resolved == legacy
    assert resolved == endpoint


@pytest.mark.unit
def test_descriptor_strips_whitespace_like_legacy_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The descriptor strips like the legacy ``.strip()`` call did."""
    monkeypatch.setenv(_ENV, "  http://phoenix:6006  ")
    assert contract_otel_exporter_endpoint() == "http://phoenix:6006"


@pytest.mark.unit
def test_descriptor_returns_empty_when_unset_opt_in_semantics() -> None:
    """Unset endpoint resolves to '' (tracing disabled) — NOT a raise.

    OTEL tracing is opt-in; absence is a valid disabled state, distinct from the
    fail-closed QDRANT/GRAPH endpoints. The descriptor returns '' so
    ``configure_tracing`` takes its existing "empty => disabled" branch.
    """
    # _clean_otel_env fixture guarantees the var is unset.
    assert contract_otel_exporter_endpoint() == ""


@pytest.mark.unit
def test_descriptor_returns_empty_for_empty_string_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly-empty env value also resolves to '' (disabled)."""
    monkeypatch.setenv(_ENV, "")
    assert contract_otel_exporter_endpoint() == ""


@pytest.mark.unit
def test_descriptor_raises_on_structurally_malformed_contract(
    tmp_path,
) -> None:
    """A contract missing the descriptor mapping is a wiring bug — raise."""
    bad = tmp_path / "bad_contract.yaml"
    bad.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="descriptor mapping"):
        contract_otel_exporter_endpoint(bad)


@pytest.mark.unit
def test_descriptor_raises_on_non_string_field(tmp_path) -> None:
    """A non-string descriptor field is a wiring bug — raise."""
    bad = tmp_path / "bad_contract.yaml"
    bad.write_text(
        "name: x\ndescriptor:\n  otel_exporter_otlp_endpoint: 123\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="string"):
        contract_otel_exporter_endpoint(bad)


@pytest.mark.unit
def test_configure_tracing_disabled_when_endpoint_unset() -> None:
    """End-to-end: configure_tracing returns False via the descriptor when unset."""
    from omnibase_infra.runtime.tracing import configure_tracing

    # _clean_otel_env fixture guarantees the endpoint var is unset.
    assert configure_tracing() is False
