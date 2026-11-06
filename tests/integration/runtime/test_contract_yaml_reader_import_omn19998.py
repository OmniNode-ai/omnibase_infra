# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Low-level contract readers must not import the handler package (OMN-19998)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_SCRIPT = """
import builtins
import sys
from pathlib import Path
import omnibase_infra.tools.contract_topic_extractor
import omnibase_infra.runtime.contract_terminal_events
assert "omnibase_infra.handlers.handler_package_announcer" not in sys.modules

# Auto-wiring's package initialization intentionally imports the announcer.
# The low-level read itself must not depend on that handler, even when loaded.
from omnibase_infra.runtime.auto_wiring.models.model_consume_concurrency import (
    load_consume_concurrency,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import load_published_events_map

original_import = builtins.__import__
def refuse_announcer_import(name, *args, **kwargs):
    if name == "omnibase_infra.handlers.handler_package_announcer":
        raise AssertionError("low-level reader imported package announcer")
    return original_import(name, *args, **kwargs)
builtins.__import__ = refuse_announcer_import

path = Path(sys.argv[1])
assert load_consume_concurrency(path).max_in_flight_records == 3
assert load_published_events_map(path) == {"ProbeCompleted": "probe-completed"}
assert omnibase_infra.runtime.contract_terminal_events.load_terminal_event_topics(
    path
) == frozenset({"probe-completed"})
assert omnibase_infra.tools.contract_topic_extractor.read_projection_api_topics(
    path
) == ("probe-snapshot",)
"""


def test_readers_import_without_handler_module(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "consume_concurrency:\n"
        "  max_in_flight_records: 3\n"
        "published_events:\n"
        "  - event_type: ProbeCompleted\n"
        "    topic: probe-completed\n"
        "terminal_event: probe-completed\n"
        "projection_api:\n"
        "  expose: true\n"
        "  bus_backed: true\n"
        "  topic: probe-snapshot\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(contract)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_reader_refuses_python_object_tags(tmp_path: Path) -> None:
    import yaml

    from omnibase_infra.contracts.contract_yaml import read_contract_yaml

    contract = tmp_path / "contract.yaml"
    contract.write_text("a: !!python/object:os.system {}\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        read_contract_yaml(contract)
