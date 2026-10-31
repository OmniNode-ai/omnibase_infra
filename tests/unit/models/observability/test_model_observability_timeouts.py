# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The observability services read their timeouts from a declared file (OMN-21623)."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.observability import (
    ModelObservabilityTimeouts,
    observability_timeouts,
)
from omnibase_infra.services.observability.agent_actions.writer_postgres import (
    WriterAgentActionsPostgres,
)
from omnibase_infra.services.observability.baselines.service_batch_compute_baselines import (
    ServiceBatchComputeBaselines,
)
from omnibase_infra.services.observability.consumer_health.writer_postgres import (
    WriterConsumerHealthPostgres,
)
from omnibase_infra.services.observability.injection_effectiveness.reader_postgres import (
    ReaderInjectionEffectivenessPostgres,
)
from omnibase_infra.services.observability.injection_effectiveness.service_batch_compute_effectiveness import (
    ServiceBatchComputeEffectivenessMetrics,
)

_TIMEOUTS_FILE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "services"
    / "observability"
    / "observability_timeouts.yaml"
)


def _declared_block() -> dict[str, object]:
    return yaml.safe_load(_TIMEOUTS_FILE.read_text(encoding="utf-8"))["timeouts"]


@pytest.mark.unit
class TestObservabilityTimeoutsContract:
    def test_loaded_timeouts_equal_the_declared_block(self) -> None:
        loaded = observability_timeouts()
        assert loaded.model_dump() == _declared_block()

    @pytest.mark.parametrize("key", sorted(_declared_block()))
    def test_block_missing_a_value_fails_to_load(
        self, key: str, tmp_path: Path
    ) -> None:
        block = _declared_block()
        del block[key]
        path = tmp_path / "timeouts.yaml"
        path.write_text(yaml.safe_dump({"timeouts": block}), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            ModelObservabilityTimeouts.from_file(path)

    def test_file_without_a_timeouts_mapping_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "timeouts.yaml"
        path.write_text("other: 1\n", encoding="utf-8")
        with pytest.raises(ProtocolConfigurationError):
            ModelObservabilityTimeouts.from_file(path)

    def test_non_positive_timeout_is_refused(self, tmp_path: Path) -> None:
        block = _declared_block()
        block["stop_seconds"] = 0
        path = tmp_path / "timeouts.yaml"
        path.write_text(yaml.safe_dump({"timeouts": block}), encoding="utf-8")
        with pytest.raises(ValueError, match="stop_seconds"):
            ModelObservabilityTimeouts.from_file(path)


@pytest.mark.unit
class TestServicesUseDeclaredTimeouts:
    def test_writer_and_reader_query_timeout_is_the_declared_value(self) -> None:
        declared = observability_timeouts().query_seconds
        assert WriterAgentActionsPostgres(MagicMock())._query_timeout == declared
        assert (
            ReaderInjectionEffectivenessPostgres(MagicMock())._query_timeout == declared
        )

    def test_writer_circuit_breaker_opens_for_the_declared_reset_timeout(self) -> None:
        declared = observability_timeouts()
        for writer in (
            WriterAgentActionsPostgres(MagicMock()),
            WriterConsumerHealthPostgres(MagicMock()),
        ):
            assert writer.circuit_breaker_reset_timeout == declared.reset_seconds

    def test_batch_compute_defaults_are_the_declared_value(self) -> None:
        declared = observability_timeouts().batch_query_seconds
        for service in (
            ServiceBatchComputeBaselines,
            ServiceBatchComputeEffectivenessMetrics,
        ):
            default = inspect.signature(service.__init__).parameters["query_timeout"]
            assert default.default == declared
