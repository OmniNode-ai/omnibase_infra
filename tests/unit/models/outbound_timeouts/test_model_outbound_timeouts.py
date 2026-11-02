# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outbound-call timeouts are read from a declared file, not from literals (OMN-21623)."""

from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.adapters.llm import (
    adapter_code_analysis_enrichment,
    adapter_code_review_analysis,
    adapter_documentation_generation,
    adapter_test_boilerplate_generation,
)
from omnibase_infra.adapters.project_tracker import (
    linear_graphql_project_tracker_adapter,
)
from omnibase_infra.adapters.ticket import adapter_ticket_service_linear
from omnibase_infra.diagnostics import bus_audit
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.outbound_timeouts import (
    ModelOutboundTimeouts,
    load_outbound_timeouts,
)

_TIMEOUTS_FILE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "models"
    / "outbound_timeouts"
    / "outbound_timeouts.yaml"
)

_TIMEOUTS = load_outbound_timeouts()
# The adapters take the model from their caller; any name serves these timeout probes.
_PROBE_MODEL = "probe-model"


def _declared_block() -> dict[str, float]:
    return cast(
        "dict[str, float]",
        yaml.safe_load(_TIMEOUTS_FILE.read_text(encoding="utf-8"))["timeouts"],
    )


def _write(tmp_path: Path, block: dict[str, float]) -> Path:
    path = tmp_path / "timeouts.yaml"
    path.write_text(yaml.safe_dump({"timeouts": block}), encoding="utf-8")
    return path


@pytest.mark.unit
class TestOutboundTimeoutsContract:
    def test_loaded_timeouts_equal_the_declared_block(self) -> None:
        assert _TIMEOUTS.model_dump() == {
            key: float(value) for key, value in _declared_block().items()
        }

    @pytest.mark.parametrize("key", sorted(_declared_block()))
    def test_block_missing_a_value_fails_to_load(
        self, key: str, tmp_path: Path
    ) -> None:
        block = _declared_block()
        del block[key]
        with pytest.raises(ValidationError) as caught:
            ModelOutboundTimeouts.from_file(_write(tmp_path, block))
        assert [(error["loc"], error["type"]) for error in caught.value.errors()] == [
            ((key,), "missing")
        ]

    def test_file_without_a_timeouts_mapping_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "timeouts.yaml"
        path.write_text("other: 1\n", encoding="utf-8")
        with pytest.raises(ProtocolConfigurationError):
            ModelOutboundTimeouts.from_file(path)

    def test_non_positive_wait_is_refused(self, tmp_path: Path) -> None:
        block = _declared_block()
        block["git_diff_seconds"] = 0
        with pytest.raises(ValueError, match="git_diff_seconds"):
            ModelOutboundTimeouts.from_file(_write(tmp_path, block))

    @pytest.mark.parametrize(
        "key", ["llm_request_seconds", "llm_documentation_request_seconds"]
    )
    def test_request_wait_above_the_transport_ceiling_is_refused(
        self, key: str, tmp_path: Path
    ) -> None:
        block = _declared_block()
        block[key] = float(block["llm_long_form_max_seconds"]) + 1
        with pytest.raises(ValueError, match=key):
            ModelOutboundTimeouts.from_file(_write(tmp_path, block))

    def test_coder_request_above_its_own_ceiling_is_refused(
        self, tmp_path: Path
    ) -> None:
        block = _declared_block()
        block["llm_coder_max_seconds"] = float(block["llm_coder_request_seconds"]) - 1
        with pytest.raises(
            ValidationError,
            match="llm_coder_request_seconds exceeds llm_coder_max_seconds",
        ):
            ModelOutboundTimeouts.from_file(_write(tmp_path, block))

    def test_adapter_imports_do_not_read_the_timeout_file(self, tmp_path: Path) -> None:
        program = """
import importlib
from pathlib import Path
original_open = Path.open

def refuse_timeout_file(path, *args, **kwargs):
    if path.name == "outbound_timeouts.yaml":
        raise FileNotFoundError("timeout file unavailable for import probe")
    return original_open(path, *args, **kwargs)

Path.open = refuse_timeout_file
for module in (
    "adapters.llm.adapter_code_analysis_enrichment",
    "adapters.llm.adapter_code_review_analysis",
    "adapters.llm.adapter_documentation_generation",
    "adapters.llm.adapter_test_boilerplate_generation",
    "adapters.project_tracker.linear_graphql_project_tracker_adapter",
    "adapters.ticket.adapter_ticket_service_linear",
    "diagnostics.bus_audit",
):
    importlib.import_module("omnibase_infra." + module)
from omnibase_infra.models.outbound_timeouts import load_outbound_timeouts
assert load_outbound_timeouts.cache_info().currsize == 0
try:
    load_outbound_timeouts()
except FileNotFoundError as error:
    assert "timeout file unavailable" in str(error)
else:
    raise AssertionError("first use must refuse a missing settings file")
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def _default(callable_: Callable[..., object], parameter: str) -> object:
    return inspect.signature(callable_).parameters[parameter].default


_TRACKER = linear_graphql_project_tracker_adapter.AdapterLinearGraphQLProjectTracker

_SITES: list[tuple[str, object, str]] = [
    (
        "linear graphql request",
        _default(_TRACKER.__init__, "timeout_seconds"),
        "linear_graphql_request_seconds",
    ),
    (
        "linear graphql transport",
        _default(_TRACKER.graphql_transport, "timeout_seconds"),
        "linear_graphql_request_seconds",
    ),
    (
        "linear graphql close",
        _default(_TRACKER.close, "timeout_seconds"),
        "close_seconds",
    ),
    (
        "linear ticket request",
        _default(adapter_ticket_service_linear.AdapterTicketLinear.__init__, "timeout"),
        "linear_ticket_request_seconds",
    ),
    (
        "linear ticket close",
        _default(
            adapter_ticket_service_linear.AdapterTicketLinear.close, "timeout_seconds"
        ),
        "close_seconds",
    ),
    (
        "bus audit poll",
        bus_audit.AuditConfig(broker="broker").poll_timeout,
        "audit_poll_seconds",
    ),
]


@pytest.mark.unit
class TestSiteDefaults:
    @pytest.mark.parametrize(
        ("site", "value", "field"), _SITES, ids=[site[0] for site in _SITES]
    )
    def test_site_defers_the_declared_default_until_use(
        self, site: str, value: object, field: str
    ) -> None:
        if site == "bus audit poll":
            assert value == getattr(_TIMEOUTS, field), site
        else:
            assert value is None, site


class _Proc:
    """Subprocess double whose ``communicate`` returns an empty diff."""

    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""


class _Metadata:
    topics: dict[str, object] = {}


@pytest.mark.unit
class TestCallSitesPassDeclaredTimeouts:
    """Sites that bind the value in the body, not in a signature default."""

    @pytest.mark.asyncio
    async def test_git_diff_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[float] = []

        async def fake_exec(*_args: object, **_kwargs: object) -> _Proc:
            return _Proc()

        async def fake_wait_for(awaitable: Any, timeout: float) -> object:
            seen.append(timeout)
            return await awaitable

        monkeypatch.setattr(
            adapter_code_analysis_enrichment.asyncio,
            "create_subprocess_exec",
            fake_exec,
        )
        monkeypatch.setattr(
            adapter_code_analysis_enrichment.asyncio, "wait_for", fake_wait_for
        )
        enrichment = adapter_code_analysis_enrichment.AdapterCodeAnalysisEnrichment
        await enrichment._read_git_diff()
        assert seen == [_TIMEOUTS.git_diff_seconds]

    def test_coder_transport_ceiling(self) -> None:
        adapter = adapter_code_analysis_enrichment.AdapterCodeAnalysisEnrichment(
            base_url="http://localhost:1", model=_PROBE_MODEL
        )
        assert (
            adapter._transport._max_timeout_seconds == _TIMEOUTS.llm_coder_max_seconds
        )

    @pytest.mark.parametrize(
        "adapter_class",
        [
            adapter_code_review_analysis.AdapterCodeReviewAnalysis,
            adapter_documentation_generation.AdapterDocumentationGeneration,
            adapter_test_boilerplate_generation.AdapterTestBoilerplateGeneration,
        ],
    )
    def test_long_form_transport_ceiling(self, adapter_class: Any) -> None:
        adapter = adapter_class(base_url="http://localhost:1", model=_PROBE_MODEL)
        assert (
            adapter._transport._max_timeout_seconds
            == _TIMEOUTS.llm_long_form_max_seconds
        )

    @pytest.mark.asyncio
    async def test_linear_circuit_reset(self) -> None:
        tracker = _TRACKER(api_key="unit-test")
        try:
            assert (
                tracker.circuit_breaker_reset_timeout
                == _TIMEOUTS.linear_graphql_circuit_reset_seconds
            )
        finally:
            await tracker.close()

    def test_bus_audit_topic_discovery_and_offset_waits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        discovery: list[float] = []
        offsets: list[float] = []

        class _StopAuditError(Exception):
            pass

        class FakeAdmin:
            def __init__(self, _config: object) -> None: ...

            def list_topics(self, timeout: float) -> object:
                discovery.append(timeout)
                raise _StopAuditError  # the discovery wait is recorded; stop here

        class _Partitions:
            partitions = {0: object()}

        class FakeConsumer:
            def __init__(self, _config: object) -> None: ...

            def get_watermark_offsets(
                self, _partition: object, timeout: float
            ) -> tuple[int, int]:
                offsets.append(timeout)
                return 0, 0

            def close(self) -> None: ...

        monkeypatch.setattr(bus_audit, "AdminClient", FakeAdmin)
        monkeypatch.setattr(bus_audit, "Consumer", FakeConsumer)
        config = bus_audit.AuditConfig(broker="broker")
        metadata = _Metadata()
        metadata.topics = {"t": _Partitions()}
        bus_audit._collect_watermarks(config, {"t"}, cast("Any", metadata))
        assert offsets == [_TIMEOUTS.audit_offsets_seconds]
        with pytest.raises(_StopAuditError):
            bus_audit.run_audit(config)
        assert discovery == [_TIMEOUTS.audit_topic_discovery_seconds]
