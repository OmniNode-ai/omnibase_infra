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

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
PIN_PRODUCERS = (
    "omnimarket-contract-pin-refresh.yml",
    "plugin-pin-cascade.yml",
)


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


def _other_workflows() -> list[Path]:
    return [
        path
        for path in sorted(WORKFLOWS.glob("*.yml"))
        if path.name not in PIN_PRODUCERS
    ]


@pytest.mark.unit
def test_no_workflow_starts_a_pin_producer() -> None:
    """No other workflow dispatches either producer, on any trigger or by API.

    A producer that only a person can start stays so only while nothing else in
    the directory starts it: not a ``workflow_run`` on its name, not a
    ``repository_dispatch`` that carries the old ``plugin-release`` type, not a
    ``gh workflow run`` or a dispatches API call that names its file.
    """
    names = set()
    for producer in PIN_PRODUCERS:
        loaded = yaml.safe_load((WORKFLOWS / producer).read_text(encoding="utf-8"))
        names.add(str(loaded["name"]))
    starters = re.compile(
        r"(?:" + "|".join(re.escape(p) for p in PIN_PRODUCERS) + r")"
        r"[^\n]*(?:dispatches|--ref)|"
        r"gh\s+workflow\s+run\s+\S*(?:"
        + "|".join(re.escape(p) for p in PIN_PRODUCERS)
        + ")|"
        r"plugin-release"
    )
    offenders = []
    for path in _other_workflows():
        text = path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        triggers = loaded.get("on", loaded.get(True)) or {}
        if isinstance(triggers, dict):
            watched = set(triggers.get("workflow_run", {}).get("workflows", []))
            if watched & names:
                offenders.append(
                    f"{path.name}: workflow_run on {sorted(watched & names)}"
                )
        if starters.search(text):
            offenders.append(f"{path.name}: names a producer as a dispatch target")
    assert not offenders, offenders


@pytest.mark.unit
def test_no_other_workflow_writes_a_pin_file() -> None:
    """The two pin files are written by their producers and by nothing else.

    A scheduled or event-driven workflow that opened a pin pull request itself
    would reopen the failure these two removals close. The pin files are the
    omnimarket contract pin and the runtime plugin pin floors.
    """
    pin_files = (
        ".github/omnimarket-contract-pin.yaml",
        "runtime_plugin_pin_floors",
    )
    writers = []
    for path in _other_workflows():
        text = path.read_text(encoding="utf-8")
        for pin_file in pin_files:
            for line in text.splitlines():
                stripped = line.strip()
                if pin_file in stripped and re.search(
                    r"(>>?\s*\S*"
                    + re.escape(pin_file)
                    + r"|git\s+add\b|sed\s+-i|tee\b)",
                    stripped,
                ):
                    writers.append(f"{path.name}: {stripped[:100]}")
    assert not writers, writers
