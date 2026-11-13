# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24206: pin producers advance only when a change needs a newer pin.

RULING 2026-11-09T21:54:25Z lane=orchestrator-31d9: a downstream repository
moves its pin on a sibling only when it needs something new from it. The
omnimarket contract pin refresh lost its six-hourly schedule, and the plugin
pin cascade lost its release-time ``repository_dispatch`` trigger; both now
run only by ``workflow_dispatch``, when a change needs a newer omnimarket
contract or a newer plugin pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _triggers(name: str) -> dict[str, object]:
    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # YAML 1.1 reads the bare key ``on`` as the boolean True.
    return loaded.get("on", loaded.get(True))


@pytest.mark.unit
def test_contract_pin_refresh_is_dispatch_only() -> None:
    assert set(_triggers("omnimarket-contract-pin-refresh.yml")) == {
        "workflow_dispatch"
    }


@pytest.mark.unit
def test_contract_pin_refresh_runs_no_cron() -> None:
    text = (WORKFLOWS / "omnimarket-contract-pin-refresh.yml").read_text(
        encoding="utf-8"
    )
    assert "cron:" not in text


@pytest.mark.unit
def test_plugin_pin_cascade_is_dispatch_only() -> None:
    assert set(_triggers("plugin-pin-cascade.yml")) == {"workflow_dispatch"}


@pytest.mark.unit
def test_plugin_pin_cascade_has_no_release_dispatch() -> None:
    assert "repository_dispatch" not in _triggers("plugin-pin-cascade.yml")
