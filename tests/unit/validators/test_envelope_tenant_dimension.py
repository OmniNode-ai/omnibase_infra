# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every producer says what tenant its event belongs to (OMN-16831, item 2).

The ruled item has two halves. The first -- *families with a tenant in scope
record it* -- is fixed per site and pinned by the seam tests beside this one.
This is the second: *families with no tenant concept record explicitly-none
rather than nothing*.

WHY THAT HALF NEEDS A GATE AT ALL
---------------------------------
``ModelEventEnvelope.tenant_id`` defaults to ``None``. So at the model level an
omission and a deliberate "this family has no tenant" are byte-identical, and
no reader, test or type checker can tell them apart. The declaration is the
only thing that distinguishes them, and a declaration nothing enforces decays
to a convention the next producer does not know about -- which is exactly how
the platform arrived here: the gateway heartbeat, the two re-materialization
helpers and the projection terminal each recorded a tenant somewhere and left
the dimension off, and nothing said so for a year.

So the gate is the mechanism, not a report. It refuses a construction of
``ModelEventEnvelope`` in this package's shipped source that does not pass
``tenant_id`` explicitly. ``tenant_id=None`` passes: that IS the declaration.

WHAT IT DELIBERATELY DOES NOT COVER, STATED RATHER THAN IMPLIED
---------------------------------------------------------------
* **Tests.** A test's envelope is a fixture, not a write path, and requiring a
  declaration there would be noise that trains people to ignore the gate.
* **The other two repos.** ``omnibase_core`` (24 construction sites) and
  ``omnimarket`` (40) have the same surface and are NOT covered here. The ruled
  item 2 names the runtime, which is this package; the other two are enumerated
  on the ticket as named follow-up rather than silently skipped.
* **Whether the recorded tenant is the RIGHT one.** This gate proves a producer
  answered the question. It cannot prove the answer is true -- that is what the
  projection authority's fail-closed verification is for, and it is the same
  honest limit every blast-radius gate in this repo carries.

There is no ``--baseline`` flag and there must never be one: OMN-18013 burned
the baselines that existed and ``no-baseline-refreeze`` refuses their return.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.validators.envelope_tenant_dimension import (
    PACKAGE_ROOT,
    findings,
)

_HEADER = "from omnibase_core.models.events.model_event_envelope import (\n    ModelEventEnvelope,\n)\n"


def _write(tmp_path: Path, body: str) -> Path:
    module = tmp_path / "producer.py"
    module.write_text(_HEADER + body, encoding="utf-8")
    return tmp_path


@pytest.mark.unit
def test_an_undeclared_construction_is_refused(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        "e = ModelEventEnvelope(payload={'a': 1})\n",
    )

    found = findings(root)

    assert len(found) == 1
    assert found[0].line == 4
    assert "tenant_id" in found[0].detail


@pytest.mark.unit
def test_an_explicit_none_is_the_declaration_and_passes(tmp_path: Path) -> None:
    """``tenant_id=None`` is a producer saying "this family has no tenant"."""
    root = _write(
        tmp_path,
        "e = ModelEventEnvelope(payload={'a': 1}, tenant_id=None)\n",
    )

    assert findings(root) == []


@pytest.mark.unit
def test_a_recorded_tenant_passes(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        "e = ModelEventEnvelope(payload={'a': 1}, tenant_id=slug)\n",
    )

    assert findings(root) == []


@pytest.mark.unit
def test_a_subscripted_construction_is_checked_too(tmp_path: Path) -> None:
    """``ModelEventEnvelope[dict[str, object]](...)`` is the common spelling.

    A checker that only matched the bare name would miss most of the real
    sites, so a positive control on the parameterized form is the difference
    between a gate and a decoration.
    """
    root = _write(
        tmp_path,
        "e = ModelEventEnvelope[dict[str, object]](payload={'a': 1})\n",
    )

    found = findings(root)

    assert len(found) == 1


@pytest.mark.unit
def test_keyword_unpacking_is_refused_rather_than_waved_through(
    tmp_path: Path,
) -> None:
    """``**kwargs`` cannot be read statically, so it fails closed.

    A checker that passed an unreadable call would hand anyone who wanted to
    skip the declaration a one-character bypass.
    """
    root = _write(
        tmp_path,
        "e = ModelEventEnvelope(**values)\n",
    )

    found = findings(root)

    assert len(found) == 1
    assert "**" in found[0].detail


@pytest.mark.unit
def test_an_unrelated_call_is_not_flagged(tmp_path: Path) -> None:
    """Negative control: the checker matches this constructor and nothing else."""
    root = _write(
        tmp_path,
        "e = SomeOtherModel(payload={'a': 1})\n",
    )

    assert findings(root) == []


@pytest.mark.unit
def test_this_package_declares_a_tenant_at_every_construction() -> None:
    """The ratchet. RED before OMN-16831 item 2; it is the gate's live proof."""
    found = findings(PACKAGE_ROOT)

    assert found == [], "\n".join(f"{f.path}:{f.line} {f.detail}" for f in found)
