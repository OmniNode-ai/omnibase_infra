# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The writer restart-bound rule, shown on manifests built here (OMN-22644).

The parity test in ``test_lab_lane_census_facts_omn20934.py`` reads the lab's
real manifest, which only exists where the private overlay is fetched. These
cases run everywhere and name each refusal: a writer that runs as a service
needs a bound tighter than the broker's, and a writer that is not bounded must
be declared ``profile_gated`` and nothing else.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.writer_restart_bound_rule import assert_writer_bound_rule

pytestmark = pytest.mark.unit

_LANE = "dev"
_WRITER = "omnimarket-projection-savings-writer"
_BROKER = "omnibase-infra-redpanda"


def _manifest(*writer_entries: dict[str, Any]) -> dict[str, Any]:
    services: list[dict[str, Any]] = [
        {"name": _BROKER, "kind": "service", "restart_bound": 5},
        *writer_entries,
    ]
    return {"lanes": {_LANE: {"services": services}}}


def _check(manifest: dict[str, Any]) -> None:
    assert_writer_bound_rule(manifest, _LANE, _WRITER, _BROKER)


def test_a_writer_running_as_a_service_passes_with_a_tighter_bound() -> None:
    _check(_manifest({"name": _WRITER, "kind": "service", "restart_bound": 2}))


def test_a_writer_running_as_a_service_fails_with_an_equal_bound() -> None:
    with pytest.raises(AssertionError, match="not tighter"):
        _check(_manifest({"name": _WRITER, "kind": "service", "restart_bound": 5}))


def test_a_writer_running_as_a_service_fails_with_a_looser_bound() -> None:
    with pytest.raises(AssertionError, match="not tighter"):
        _check(_manifest({"name": _WRITER, "kind": "service", "restart_bound": 9}))


def test_a_writer_running_as_a_service_fails_with_no_bound() -> None:
    with pytest.raises(AssertionError, match="not 'profile_gated'"):
        _check(_manifest({"name": _WRITER, "kind": "service"}))


def test_a_folded_writer_declared_profile_gated_passes() -> None:
    _check(_manifest({"name": _WRITER, "kind": "profile_gated"}))


def test_a_writer_dropped_from_the_manifest_fails() -> None:
    with pytest.raises(AssertionError, match="neither bounded nor declared"):
        _check(_manifest())


def test_a_writer_declared_any_other_kind_without_a_bound_fails() -> None:
    with pytest.raises(AssertionError, match="not 'profile_gated'"):
        _check(_manifest({"name": _WRITER, "kind": "keepalive"}))
