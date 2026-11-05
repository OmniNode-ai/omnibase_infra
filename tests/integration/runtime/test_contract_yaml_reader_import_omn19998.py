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
import sys
import omnibase_infra.tools.contract_topic_extractor
import omnibase_infra.runtime.contract_terminal_events
assert "omnibase_infra.handlers.handler_package_announcer" not in sys.modules
"""


def test_readers_import_without_handler_module() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_reader_refuses_python_object_tags(tmp_path: Path) -> None:
    import yaml

    from omnibase_infra.handlers.handler_package_announcer import read_contract_yaml

    contract = tmp_path / "contract.yaml"
    contract.write_text("a: !!python/object:os.system {}\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        read_contract_yaml(contract)
