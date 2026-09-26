# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``discover_contracts()`` does not re-parse unchanged contracts (OMN-19373).

WHY THIS EXISTS. ``omninode-runtime-effects`` re-runs a full contract-discovery
sweep every ~316s. Measured inside the deployed pod on onex-dev 2026-09-26,
``discover_contracts()`` took 24.036s and an immediate second call took
23.922s -- there was no memoization at all, and ``_parse_contract`` over 544
entry points accounted for 24.099s of it. ``entry_points()`` cost 0.056s and
``_resolve_contract_path`` 0.052s, so essentially the whole sweep is YAML
re-parsing of files that cannot change inside a container image.

WHAT THAT COST. A gateway heartbeat issued during a sweep is answered only as
the sweep ends: on the 22-beat readback of 2026-09-26T15:46Z, beat 6 returned
200 after 15.333s and the server-recorded gap was 25.103s, against the
OMN-15957 harness tolerance of ``heartbeat_interval + 5s`` = 20s. That is why
Test 3 fails while every beat is a 200 -- the sweep no longer refuses the
request, it delays it past the acceptance bar.

WHAT THESE TESTS HOLD. That a repeat scan whose entry points and contract files
are byte-unchanged reuses the previous manifest instead of re-parsing, and that
any change to a contract file, to the entry-point set, or to the opt-out
environment variable still forces a full re-parse. They do NOT assert a
duration -- a timing assertion would be flaky on a shared runner. They count
``_parse_contract`` calls, which is the quantity the sweep's cost is made of.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring import discovery as discovery_mod
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts

_EP_MODULE = "omnibase_infra.runtime.auto_wiring.discovery.entry_points"

_CONTRACT = """\
name: "{name}"
node_type: "EFFECT_GENERIC"
contract_version:
  major: 1
  minor: 2
  patch: 3
node_version: "2.0.0"
description: "A test node"
"""


@pytest.fixture(autouse=True)
def _clear_cache():
    """The memo is module state; no test may inherit another test's."""
    clear = getattr(discovery_mod, "discover_contracts_cache_clear", None)
    if clear is not None:
        clear()
    yield
    if clear is not None:
        clear()


def _entry_point(name: str, *, dist_name: str = "my-plugin") -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.dist = MagicMock()
    ep.dist.name = dist_name
    ep.dist.version = "1.0.0"
    ep.load.return_value = type("FakeNode", (), {"process": lambda self: None})
    return ep


class _ParseCounter:
    """Wraps ``_parse_contract`` and counts how often it actually parses."""

    def __init__(self) -> None:
        self._real = discovery_mod._parse_contract
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return self._real(*args, **kwargs)


def _fixture_on_disk(tmp_path: Path, name: str = "my_effect") -> tuple[Path, Path]:
    contract = tmp_path / "contract.yaml"
    contract.write_text(_CONTRACT.format(name=name))
    module_file = tmp_path / "node.py"
    module_file.write_text("")
    return contract, module_file


@pytest.mark.unit
def test_the_counter_sees_the_first_scan(tmp_path: Path) -> None:
    """Positive control.

    Every assertion below is about a parse count NOT growing. If the counter
    were never wired to anything, a zero would satisfy those vacuously. This
    proves the first scan is counted, so a later zero means "did not re-parse"
    rather than "was never measured".
    """
    _, module_file = _fixture_on_disk(tmp_path)
    ep = _entry_point("my_effect")
    counter = _ParseCounter()

    with (
        patch(_EP_MODULE, return_value=[ep]),
        patch("inspect.getfile", return_value=str(module_file)),
        patch.object(discovery_mod, "_parse_contract", counter),
    ):
        manifest = discover_contracts()

    assert manifest.total_discovered == 1
    assert counter.calls == 1


@pytest.mark.unit
def test_a_second_scan_does_not_reparse_unchanged_contracts(tmp_path: Path) -> None:
    """The sweep's whole cost. Red against today's code, which re-parses."""
    _, module_file = _fixture_on_disk(tmp_path)
    ep = _entry_point("my_effect")
    counter = _ParseCounter()

    with (
        patch(_EP_MODULE, return_value=[ep]),
        patch("inspect.getfile", return_value=str(module_file)),
        patch.object(discovery_mod, "_parse_contract", counter),
    ):
        first = discover_contracts()
        after_first = counter.calls
        second = discover_contracts()

    assert after_first == 1, "first scan must parse"
    assert counter.calls == after_first, (
        "the second scan re-parsed the contract although nothing changed; "
        f"parse calls went {after_first} -> {counter.calls}"
    )
    assert second.total_discovered == first.total_discovered
    assert [c.name for c in second.contracts] == [c.name for c in first.contracts]


@pytest.mark.unit
def test_editing_a_contract_forces_a_reparse(tmp_path: Path) -> None:
    """A cache that cannot notice an edit is a correctness bug, not a speedup."""
    contract, module_file = _fixture_on_disk(tmp_path)
    ep = _entry_point("my_effect")
    counter = _ParseCounter()

    with (
        patch(_EP_MODULE, return_value=[ep]),
        patch("inspect.getfile", return_value=str(module_file)),
        patch.object(discovery_mod, "_parse_contract", counter),
    ):
        discover_contracts()
        after_first = counter.calls
        # A different name AND a different length, so neither an mtime-only nor
        # a size-only fingerprint can miss it.
        contract.write_text(_CONTRACT.format(name="my_effect_renamed_longer"))
        os.utime(contract, (1_000_000, 1_000_000))
        manifest = discover_contracts()

    assert counter.calls == after_first + 1
    assert [c.name for c in manifest.contracts] == ["my_effect_renamed_longer"]


@pytest.mark.unit
def test_a_changed_entry_point_set_forces_a_reparse(tmp_path: Path) -> None:
    """Installing or removing a node package must not be served from the memo."""
    _, module_file = _fixture_on_disk(tmp_path)
    counter = _ParseCounter()

    with (
        patch("inspect.getfile", return_value=str(module_file)),
        patch.object(discovery_mod, "_parse_contract", counter),
    ):
        with patch(_EP_MODULE, return_value=[_entry_point("my_effect")]):
            discover_contracts()
        after_first = counter.calls
        with patch(
            _EP_MODULE,
            return_value=[_entry_point("my_effect", dist_name="other-plugin")],
        ):
            discover_contracts()

    assert counter.calls == after_first + 1


@pytest.mark.unit
def test_the_memo_can_be_switched_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator must be able to get the old behaviour back without a deploy."""
    monkeypatch.setenv("ONEX_AUTOWIRING_DISCOVERY_CACHE", "0")
    _, module_file = _fixture_on_disk(tmp_path)
    ep = _entry_point("my_effect")
    counter = _ParseCounter()

    with (
        patch(_EP_MODULE, return_value=[ep]),
        patch("inspect.getfile", return_value=str(module_file)),
        patch.object(discovery_mod, "_parse_contract", counter),
    ):
        discover_contracts()
        after_first = counter.calls
        discover_contracts()

    assert counter.calls == after_first + 1


@pytest.mark.unit
def test_a_vanished_contract_file_forces_a_reparse(tmp_path: Path) -> None:
    """A missing file must invalidate rather than be silently treated as equal."""
    contract, module_file = _fixture_on_disk(tmp_path)
    ep = _entry_point("my_effect")
    counter = _ParseCounter()

    with (
        patch(_EP_MODULE, return_value=[ep]),
        patch("inspect.getfile", return_value=str(module_file)),
        patch.object(discovery_mod, "_parse_contract", counter),
    ):
        discover_contracts()
        after_first = counter.calls
        contract.unlink()
        manifest = discover_contracts()

    assert counter.calls == after_first
    assert manifest.total_discovered == 0
    assert manifest.total_errors == 1
