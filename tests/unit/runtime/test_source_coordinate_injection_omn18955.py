# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18955 — the runtime must hand projection writers the source coordinates.

WHAT WAS BROKEN
---------------
A projection writer under in-process dispatch builds its snapshot-delta
``MessageMeta`` from two injected payload keys, ``_partition`` and ``_offset``.
The runtime injected neither, so every writer published every delta at
partition 0 / offset 0. The consuming ``SnapshotCache`` drops a delta whose
``source_offset`` does not exceed the cached one for the same source topic and
partition, so every delta after the FIRST for a given key was discarded as an
idempotent replay -- first writer wins forever.

The exposure then freezes while sitting at lag ZERO, because the consumer is
reading everything and applying nothing. Measured on the .201 dev lane before
this change: 6,210,195 lifetime drops on the consumer-flow exposure, 36,087 on
runner-fleet, both at ``lag: 0`` with ``applied_offset == end_offset``, and
``GET /ready`` answering 503 while ``GET /projections`` answered 200.

That 503 is the readiness endpoint being CORRECT. The consumer's drop is also
correct -- it is a real idempotent-replay guard. The producer was the wrong
half, and this is the producer half (the consumer half landed as OMN-18905).

WHY A CONTEXT CHANNEL
---------------------
``ModelEventEnvelope`` declares no partition or offset field and is
``extra="forbid"``, so the pair cannot ride it without a core release. The
coordinates DO exist on ``ModelEventMessage`` straight off the aiokafka record;
they are lost when the consume callback rebuilds the envelope from
``message.value`` alone. A task-local channel bound in the frame that still
holds the record carries them the rest of the way.

WHAT THIS MODULE PINS
---------------------
Both directions, because only one of them is the defect and only the other one
proves the fix did not simply hardcode a zero: coordinates are injected when the
record carries them, and are ABSENT when it does not. An absent key is the
correct terminal state for a record with no coordinates; a defaulted zero is the
defect itself.
"""

from __future__ import annotations

from typing import Final

import pytest

from omnibase_infra.runtime.dispatch_envelope_context import (
    bind_source_coordinate,
    current_source_coordinate,
)

pytestmark = pytest.mark.unit

_PARTITION_KEY: Final[str] = "_partition"
_OFFSET_KEY: Final[str] = "_offset"


class _Record:
    """The shape the transport hands the consume boundary."""

    def __init__(self, partition: object, offset: object) -> None:
        self.partition = partition
        self.offset = offset


class _Bare:
    """A record-like object carrying no coordinates at all."""


class TestTheChannelBindsOnlyARealPair:
    def test_nothing_is_bound_before_any_bind(self) -> None:
        # Guards every assertion below: a leaked value from another test would
        # make the negative cases pass for the wrong reason.
        assert current_source_coordinate() is None

    def test_a_real_pair_is_bound_and_visible(self) -> None:
        with bind_source_coordinate(_Record(partition=3, offset="4171")):
            assert current_source_coordinate() == (3, "4171")

    def test_partition_zero_and_offset_zero_are_a_real_pair(self) -> None:
        # The defect was a DEFAULTED zero, never a measured one. Partition 0 is
        # the only partition a single-partition topic has, and offset 0 is a
        # real first record; refusing them would break the common case.
        with bind_source_coordinate(_Record(partition=0, offset="0")):
            assert current_source_coordinate() == (0, "0")

    def test_an_integer_offset_is_accepted_and_carried_as_text(self) -> None:
        # The transport model declares offset as str; an int is coerced once
        # here rather than at two readers that could then disagree.
        with bind_source_coordinate(_Record(partition=1, offset=99)):
            assert current_source_coordinate() == (1, "99")

    @pytest.mark.parametrize(
        ("partition", "offset"),
        [
            (None, "7"),
            (2, None),
            (None, None),
            ("2", "7"),
            (True, "7"),
        ],
        ids=[
            "no-partition",
            "no-offset",
            "neither",
            "text-partition",
            "bool-partition",
        ],
    )
    def test_an_incomplete_or_mistyped_pair_binds_nothing(
        self, partition: object, offset: object
    ) -> None:
        # Absent is a statement. Binding half a pair, or a bool that int()
        # would silently accept as 0/1, would put a fabricated coordinate on
        # the wire -- which is the defect, not the fix.
        with bind_source_coordinate(_Record(partition=partition, offset=offset)):
            assert current_source_coordinate() is None

    def test_an_object_with_no_coordinate_attributes_binds_nothing(self) -> None:
        with bind_source_coordinate(_Bare()):
            assert current_source_coordinate() is None

    def test_the_binding_is_restored_on_exit(self) -> None:
        with bind_source_coordinate(_Record(partition=5, offset="50")):
            assert current_source_coordinate() == (5, "50")
        assert current_source_coordinate() is None

    def test_the_binding_is_restored_when_the_body_raises(self) -> None:
        # A dispatch that raises must not leave a stale coordinate bound for
        # the next message on the same task.
        with pytest.raises(RuntimeError):
            with bind_source_coordinate(_Record(partition=6, offset="60")):
                raise RuntimeError("handler failed")
        assert current_source_coordinate() is None

    def test_an_inner_binding_restores_the_outer_one(self) -> None:
        with bind_source_coordinate(_Record(partition=1, offset="1")):
            with bind_source_coordinate(_Record(partition=2, offset="2")):
                assert current_source_coordinate() == (2, "2")
            assert current_source_coordinate() == (1, "1")


class TestTheProjectionSiteInjectsFromTheChannel:
    """The wiring half, asserted against the source rather than a live broker.

    A full dispatch would need a bus, an engine and a database adapter. What
    has to be true is narrower and is exactly what regressed: the projection
    dispatch site reads the channel and injects BOTH keys, guarded so that an
    unbound channel injects neither.
    """

    @staticmethod
    def _wiring_source() -> str:
        from pathlib import Path

        import omnibase_infra.runtime.auto_wiring.handler_wiring as wiring

        return Path(wiring.__file__).read_text(encoding="utf-8")

    def test_the_site_injects_both_keys(self) -> None:
        source = self._wiring_source()
        assert f'input_data["{_PARTITION_KEY}"] = source_coordinate[0]' in source
        assert f'input_data["{_OFFSET_KEY}"] = source_coordinate[1]' in source

    def test_the_injection_is_guarded_on_the_channel(self) -> None:
        source = self._wiring_source()
        assert "source_coordinate = current_source_coordinate()" in source
        assert "if source_coordinate is not None:" in source

    def test_every_consume_boundary_binds_the_channel(self) -> None:
        # Three boundaries reach the same dispatch engine. A binding installed
        # at only one of them fixes only the lanes that happen to use it, and
        # the symptom of missing one is a silently frozen exposure rather than
        # an error -- so the count is asserted, not eyeballed.
        from pathlib import Path

        import omnibase_infra.runtime.event_bus_subcontract_wiring as subcontract

        wiring_binds = self._wiring_source().count("with bind_source_coordinate(")
        assert wiring_binds == 2, (
            "handler_wiring must bind the source coordinate on BOTH its normal "
            f"and raw projection consume boundaries, found {wiring_binds}"
        )
        subcontract_source = Path(subcontract.__file__).read_text(encoding="utf-8")
        assert "with bind_source_coordinate(" in subcontract_source, (
            "the subcontract wiring is a second consumer boundary onto the same "
            "dispatch engine; without the bind, projections wired through the "
            "runtime host get no coordinates and freeze exactly as before"
        )

    def test_the_source_check_is_pointed_at_real_modules(self) -> None:
        # Positive control: the three assertions above are string reads, and a
        # wrong path would make them fail loudly rather than pass — but an
        # empty read would make the count assertion fail confusingly instead.
        source = self._wiring_source()
        assert len(source) > 10_000
        assert "def _make_projection_dispatch_callback" in source
