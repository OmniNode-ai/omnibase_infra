# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18852: the ``consume_concurrency`` contract key and how it is read.

Two properties carry the change's safety:

* **Absent means serial.** Every node in the fleet omits this key today, so
  the default has to reproduce the pre-OMN-18852 behaviour exactly -- and
  ``is_serial`` is what selects the unchanged inline ``await`` branch of
  ``EventBusKafka._consume_loop`` rather than a semaphore of size one.
* **Malformed does NOT mean absent.** An operator who believes a bound is in
  force while the lane runs serial is the position this ticket started from,
  so an unreadable declaration fails wiring instead of degrading to 1.

A third is a premise the whole approach rests on and is therefore asserted
rather than assumed: the runtime's own contract parser must tolerate this
unknown top-level key. ``discovery._parse_contract`` reads the YAML with
``yaml.safe_load`` and hand-picks the keys it knows, so an unrecognised one
never reaches ``ModelDiscoveredContract``'s ``extra="forbid"``. If that ever
stops being true, this key stops working, silently, for every node.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.runtime.auto_wiring.models.model_consume_concurrency import (
    ModelConsumeConcurrency,
    load_consume_concurrency,
)

pytestmark = pytest.mark.unit


def _write_contract(tmp_path: Path, body: str) -> Path:
    contract = tmp_path / "contract.yaml"
    contract.write_text(body, encoding="utf-8")
    return contract


class TestLoadConsumeConcurrency:
    def test_absent_key_is_serial(self, tmp_path: Path) -> None:
        """The default must be the unchanged inline path, not merely a small N."""
        path = _write_contract(tmp_path, "name: node_x\nnode_type: EFFECT_GENERIC\n")
        loaded = load_consume_concurrency(path)
        assert loaded.max_in_flight_records == 1
        assert loaded.is_serial is True

    def test_declared_bound_is_read(self, tmp_path: Path) -> None:
        path = _write_contract(
            tmp_path,
            "name: node_x\nconsume_concurrency:\n  max_in_flight_records: 4\n",
        )
        loaded = load_consume_concurrency(path)
        assert loaded.max_in_flight_records == 4
        assert loaded.is_serial is False

    def test_explicit_one_is_serial(self, tmp_path: Path) -> None:
        path = _write_contract(
            tmp_path,
            "name: node_x\nconsume_concurrency:\n  max_in_flight_records: 1\n",
        )
        assert load_consume_concurrency(path).is_serial is True

    def test_a_non_mapping_declaration_is_refused_not_defaulted(
        self, tmp_path: Path
    ) -> None:
        """``consume_concurrency: 4`` is a bound the author believes they set."""
        path = _write_contract(tmp_path, "name: node_x\nconsume_concurrency: 4\n")
        with pytest.raises(ValueError, match="not a mapping"):
            load_consume_concurrency(path)

    def test_an_unknown_subkey_is_refused(self, tmp_path: Path) -> None:
        """A typo'd field name must not silently leave the bound at 1."""
        path = _write_contract(
            tmp_path,
            "name: node_x\nconsume_concurrency:\n  max_inflight_records: 4\n",
        )
        with pytest.raises(ValidationError):
            load_consume_concurrency(path)

    def test_zero_is_refused(self, tmp_path: Path) -> None:
        """A bound that admits nothing is a wedged consumer."""
        path = _write_contract(
            tmp_path,
            "name: node_x\nconsume_concurrency:\n  max_in_flight_records: 0\n",
        )
        with pytest.raises(ValidationError):
            load_consume_concurrency(path)

    def test_an_unbounded_declaration_is_refused(self, tmp_path: Path) -> None:
        """Each in-flight record holds a handler and its provider connection."""
        path = _write_contract(
            tmp_path,
            "name: node_x\nconsume_concurrency:\n  max_in_flight_records: 10000\n",
        )
        with pytest.raises(ValidationError):
            load_consume_concurrency(path)

    def test_a_non_mapping_contract_is_refused(self, tmp_path: Path) -> None:
        path = _write_contract(tmp_path, "- not\n- a mapping\n")
        with pytest.raises(ValueError, match="must contain a mapping"):
            load_consume_concurrency(path)

    def test_a_missing_file_is_serial_not_a_refusal(self, tmp_path: Path) -> None:
        """A contract that never reached disk carries no declaration to read.

        Auto-wiring calls this with synthetic and non-disk-sourced contracts
        whose ``contract_path`` points at nothing; refusing there would fail
        wiring for every one of them. This is the ``load_published_events_map``
        disposition, and it is distinct from the malformed case above.
        """
        loaded = load_consume_concurrency(tmp_path / "does-not-exist.yaml")
        assert loaded.is_serial is True


class TestModelConsumeConcurrency:
    def test_is_frozen(self) -> None:
        model = ModelConsumeConcurrency(max_in_flight_records=2)
        with pytest.raises(ValidationError):
            model.max_in_flight_records = 3  # type: ignore[misc]


class TestContractParserToleratesTheKey:
    """The premise: an unknown top-level key must not break auto-wiring.

    Asserted against the real parser rather than reasoned about, because if
    it ever changes the key stops working with no error anywhere.
    """

    def test_parse_contract_ignores_the_unknown_top_level_key(
        self, tmp_path: Path
    ) -> None:
        from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract

        body = {
            "name": "node_omn18852_probe",
            "node_type": "EFFECT_GENERIC",
            "description": "fixture",
            "contract_version": "1.0.0",
            "consume_concurrency": {"max_in_flight_records": 4},
        }
        path = tmp_path / "contract.yaml"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")

        parsed = _parse_contract(
            contract_path=path,
            entry_point_name="node_omn18852_probe",
            package_name="omnibase_infra",
            package_version="0.0.0",
        )
        assert parsed.name == "node_omn18852_probe"
        # The key is invisible to the typed model and readable from the file.
        assert not hasattr(parsed, "consume_concurrency")
        assert load_consume_concurrency(parsed.contract_path).max_in_flight_records == 4
