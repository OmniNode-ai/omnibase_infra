# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Off-registry mode for the omnimarket drift guard (OMN-17255).

The guard's reference point has always been the canonical clone at
``$OMNI_HOME/omnimarket``. On a customer's machine there is no such clone, so
``canonical_local_omnimarket_commit`` returned ``None`` and the guard returned
**silently** -- exit 0, zero drift lines, zero skip lines. Measured on
2026-09-18 (row L2 of the local-path ground truth): ``env -u OMNI_HOME onex
delegate "say ok"`` exited 0 with nothing at all on stderr.

A silent pass is indistinguishable from a check that never ran
(omni_home/CLAUDE.md rule 8, memory ``feedback_no_defensive_no_defaults``).
These tests pin the replacement: off-registry the guard compares the installed
omni-internal layer against the pins PACKAGED INSIDE the installed artifacts,
and it emits exactly one structured line on stderr for **every** verdict --
IN_SYNC, DRIFTED and SKIPPED alike. The line is the deliverable; the exit code
alone never was.
"""

from __future__ import annotations

import logging

import pytest

from omnibase_infra.cli import omnimarket_drift_guard as guard
from omnibase_infra.cli.omnimarket_drift_guard import (
    EnumOffRegistryReason,
    EnumOffRegistryVerdict,
    check_omnimarket_drift,
    resolve_off_registry_check,
)

pytestmark = pytest.mark.unit

_FAKE_SHA = "a" * 40

#: One installed omnimarket whose packaged requirements are satisfied by the
#: rest of the fake environment. Shaped exactly like the real thing: the
#: ``Requires-Dist`` entries ARE the ``[project].dependencies`` the installer
#: packaged into the wheel.
_HEALTHY_ENV: dict[str, tuple[str, tuple[str, ...]]] = {
    "omnimarket": (
        "0.4.121",
        (
            "omnibase-compat==0.5.7",
            "omnibase-core<0.48.0,>=0.47.14",
            "omnibase-infra<0.39.0,>=0.38.31",
            "omnibase-spi<0.24.0,>=0.23.3",
            "omninode-memory==0.18.0",
            "anthropic>=0.40.0",
        ),
    ),
    "omnibase-infra": ("0.38.33", ("omnibase-compat==0.5.7", "omnibase-core==0.47.17")),
    "omnibase-core": ("0.47.17", ()),
    "omnibase-spi": ("0.23.3", ()),
    "omnibase-compat": ("0.5.7", ()),
    "omninode-memory": ("0.18.0", ()),
    "anthropic": ("0.40.1", ()),
}

#: The live shape measured on this host on 2026-09-18: omnimarket 0.4.121
#: declares ``omnibase-infra>=0.38.31`` and 0.38.30 is what is installed. The
#: pre-OMN-17255 guard could not see this at all off-registry.
_DRIFTED_ENV = dict(_HEALTHY_ENV) | {"omnibase-infra": ("0.38.30", ())}


def _bind_env(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, tuple[str, tuple[str, ...]]],
) -> None:
    """Bind a fake installed environment at the guard's single metadata seam."""

    def _metadata(name: str) -> tuple[str, tuple[str, ...]] | None:
        return environment.get(guard.canonical_distribution_name(name))

    monkeypatch.setattr(guard, "installed_distribution_metadata", _metadata)


def _off_registry_lines(captured: str) -> list[str]:
    return [line for line in captured.splitlines() if line.startswith("drift_guard:")]


def _fields(line: str) -> dict[str, str]:
    body = line.split("drift_guard:", 1)[1].strip()
    return dict(token.split("=", 1) for token in body.split(" "))


# --------------------------------------------------------------------------- #
# Q-4: the explicit verdict line exists, for every verdict
# --------------------------------------------------------------------------- #


def test_off_registry_in_sync_emits_one_verdict_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Workspace unset + packaged pins present and satisfied -> off-registry
    mode with a verdict line. Exit behaviour (return, no raise) is unchanged."""
    _bind_env(monkeypatch, _HEALTHY_ENV)

    check = check_omnimarket_drift(omni_home=None)

    assert check is not None
    assert check.verdict is EnumOffRegistryVerdict.IN_SYNC
    lines = _off_registry_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    fields = _fields(lines[0])
    assert fields["mode"] == "off-registry"
    assert fields["verdict"] == "IN_SYNC"
    assert fields["reason"] == EnumOffRegistryReason.PACKAGED_PINS_SATISFIED.value
    assert fields["unsatisfied"] == "0"
    assert fields["omnimarket"] == "0.4.121"
    assert fields["anchor"] == "omnimarket@0.4.121"


def test_off_registry_drift_is_reported_and_does_NOT_block_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The live 2026-09-18 shape: the installed omnibase-infra is BELOW the
    floor the installed omnimarket wheel declares.

    Operator ruling, 2026-09-18 (the OMN-17255 re-scope under the local-path
    goal, row L2): off registry the guard must not block dispatch. The two
    positions that look opposed are not about the same thing -- the goal needs
    the guard not to REFUSE on a machine with no clone, this ticket needs it
    not to be SILENTLY ABSENT. So: the verdict is DRIFTED, it is stated in
    full, and the call returns. Blocking would also be a remedy nobody on that
    machine can apply.
    """
    _bind_env(monkeypatch, _DRIFTED_ENV)

    with caplog.at_level(logging.WARNING):
        check = check_omnimarket_drift(omni_home=None)  # must NOT raise

    assert check is not None
    assert check.verdict is EnumOffRegistryVerdict.DRIFTED
    lines = _off_registry_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    fields = _fields(lines[0])
    assert fields["verdict"] == "DRIFTED"
    assert fields["reason"] == EnumOffRegistryReason.PACKAGED_PIN_UNSATISFIED.value
    assert fields["pin"] == "omnibase-infra"
    assert fields["installed"] == "0.38.30"
    assert ">=0.38.31" in fields["expected"]
    assert fields["unsatisfied"] == "1"

    # The prose half is best-effort (a library logger can have no handler),
    # which is why the structured line above carries the same facts.
    detail = caplog.text
    assert "omnibase-infra" in detail
    assert "0.38.30" in detail
    # It stays actionable off-registry: no canonical clone is named, because
    # there is none here, and a pointer to a path the reader does not have is
    # how a guard teaches people to ignore it.
    assert "$OMNIBASE_PATH/omnimarket" not in detail


@pytest.mark.parametrize(
    "environment",
    [_HEALTHY_ENV, _DRIFTED_ENV, {}],
    ids=["in_sync", "drifted", "nothing_to_compare"],
)
def test_off_registry_never_raises_for_any_verdict(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    environment: dict[str, tuple[str, tuple[str, ...]]],
) -> None:
    """The goal-row invariant, pinned on its own so a later change that makes
    one verdict blocking is RED rather than a review catch: off registry, the
    guard reports and NEVER refuses."""
    _bind_env(monkeypatch, environment)

    check = check_omnimarket_drift(omni_home=None)

    assert check is not None
    assert len(_off_registry_lines(capsys.readouterr().err)) == 1


def test_off_registry_skip_line_is_emitted_when_nothing_can_be_compared(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No packaged pin anywhere -> SKIPPED. Exit 0, as before; but the line is
    ALWAYS emitted, which is the whole point of the ticket. Asserting the line,
    not merely the exit code, is what makes a future silent regression RED."""
    _bind_env(monkeypatch, {})

    check = check_omnimarket_drift(omni_home=None)

    assert check is not None
    assert check.verdict is EnumOffRegistryVerdict.SKIPPED
    lines = _off_registry_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    fields = _fields(lines[0])
    assert fields["verdict"] == "SKIPPED"
    assert fields["reason"] == EnumOffRegistryReason.NO_PACKAGED_PIN_ANCHOR.value
    assert fields["omnimarket"] == "ABSENT"
    assert fields["anchor"] == "NONE"
    assert fields["pins"] == "0"


def test_off_registry_skip_names_an_anchor_that_declares_no_omni_pins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An anchor that IS installed but declares no omni-internal requirement is
    a different skip from an absent one, and the reason token says which."""
    _bind_env(
        monkeypatch,
        {
            "omnimarket": ("0.4.121", ("anthropic>=0.40.0",)),
            "omnibase-infra": ("0.38.33", ("click>=8.3.3",)),
            "anthropic": ("0.40.1", ()),
        },
    )

    check = check_omnimarket_drift(omni_home=None)

    assert check is not None
    assert check.verdict is EnumOffRegistryVerdict.SKIPPED
    fields = _fields(_off_registry_lines(capsys.readouterr().err)[0])
    assert fields["reason"] == EnumOffRegistryReason.NO_APPLICABLE_PACKAGED_PINS.value
    assert fields["omnimarket"] == "0.4.121"


# --------------------------------------------------------------------------- #
# Positive control: the canonical-clone path is untouched
# --------------------------------------------------------------------------- #


def test_workspace_set_keeps_todays_behaviour_and_emits_no_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positive control. With a canonical clone resolvable, the guard takes the
    commit-comparison path exactly as before: no off-registry line at all, and
    no return value to mistake for one."""
    _bind_env(monkeypatch, _DRIFTED_ENV)
    monkeypatch.setattr(
        guard, "canonical_local_omnimarket_commit", lambda omni_home=None: _FAKE_SHA
    )
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: _FAKE_SHA)
    monkeypatch.setattr(
        guard,
        "canonical_clone_attachment",
        lambda omni_home=None: guard.CanonicalCloneAttachment.ATTACHED,
    )

    assert check_omnimarket_drift(omni_home="/nonexistent-workspace") is None
    assert _off_registry_lines(capsys.readouterr().err) == []


# --------------------------------------------------------------------------- #
# The override, and the reconciler that must not run here
# --------------------------------------------------------------------------- #


def test_the_override_changes_nothing_off_registry_because_nothing_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``allow_drift`` is not consulted off-registry: there is nothing to
    override. Setting it must not suppress the line either -- an override that
    silenced the evidence would restore the exact silence this mode removes.
    """
    _bind_env(monkeypatch, _DRIFTED_ENV)

    without = check_omnimarket_drift(omni_home=None)
    without_line = _off_registry_lines(capsys.readouterr().err)
    with_override = check_omnimarket_drift(omni_home=None, allow_drift=True)
    with_line = _off_registry_lines(capsys.readouterr().err)

    assert without == with_override
    assert without_line == with_line
    assert len(with_line) == 1
    assert _fields(with_line[0])["verdict"] == "DRIFTED"


def test_off_registry_never_invokes_the_reconciler(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bound reconciler repairs a venv AGAINST the canonical clone. Off
    registry there is no clone, so running it would burn an install and then
    refuse anyway -- the same reasoning the detached-HEAD branch records."""
    _bind_env(monkeypatch, _DRIFTED_ENV)
    calls: list[int] = []

    def _reconcile() -> None:
        calls.append(1)
        raise AssertionError("the reconciler must not run off-registry")

    check = check_omnimarket_drift(omni_home=None, reconcile=_reconcile)

    assert check is not None
    assert check.verdict is EnumOffRegistryVerdict.DRIFTED
    assert calls == []


# --------------------------------------------------------------------------- #
# The resolver in isolation
# --------------------------------------------------------------------------- #


def test_anchor_falls_back_to_omnibase_infra_when_omnimarket_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolution order, stated: omnimarket's packaged pins first, then
    omnibase_infra's own. The guard ships inside omnibase_infra, so the
    fallback anchor is present by construction whenever the guard runs."""
    _bind_env(
        monkeypatch,
        {
            "omnibase-infra": ("0.38.33", ("omnibase-core==0.47.17",)),
            "omnibase-core": ("0.47.17", ()),
        },
    )

    check = resolve_off_registry_check()

    assert check.anchor == "omnibase-infra"
    assert check.omnimarket_version is None
    assert check.verdict is EnumOffRegistryVerdict.IN_SYNC


def test_an_absent_pinned_distribution_is_drift_not_a_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: a requirement whose distribution is not installed at all
    cannot be reported as satisfied."""
    _bind_env(
        monkeypatch,
        {"omnimarket": ("0.4.121", ("omnibase-compat==0.5.7",))},
    )

    check = resolve_off_registry_check()

    assert check.verdict is EnumOffRegistryVerdict.DRIFTED
    assert check.pin_name == "omnibase-compat"
    assert check.installed is None
    assert "ABSENT" in check.line


def test_the_line_is_parseable_and_carries_every_declared_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line is machine-read, so its field set is part of the contract."""
    _bind_env(monkeypatch, _HEALTHY_ENV)

    fields = _fields(resolve_off_registry_check().line)

    assert set(fields) == {
        "mode",
        "omnimarket",
        "anchor",
        "pin",
        "expected",
        "installed",
        "pins",
        "unsatisfied",
        "verdict",
        "reason",
    }
    assert " " not in "".join(fields.values())


# --------------------------------------------------------------------------- #
# The receipt half: the verdict outlives the terminal
# --------------------------------------------------------------------------- #


def test_receipt_block_is_present_off_registry_and_absent_on_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stderr line is gone once the terminal scrolls, so the same facts go
    into the run receipt -- rendered from the one object, so the two cannot
    disagree. On a registry machine the guard returns None and the receipt is
    byte-identical to today's, which is what keeps this additive.
    """
    from omnibase_infra.cli.cli_delegate import _drift_guard_receipt_block

    assert _drift_guard_receipt_block(None) == {}

    _bind_env(monkeypatch, _DRIFTED_ENV)
    block = _drift_guard_receipt_block(resolve_off_registry_check())

    assert set(block) == {"drift_guard"}
    fields = block["drift_guard"]
    assert isinstance(fields, dict)
    assert fields["mode"] == "off-registry"
    assert fields["verdict"] == EnumOffRegistryVerdict.DRIFTED.value
    assert fields["pin"] == "omnibase-infra"
    assert fields["unsatisfied"] == ["omnibase-infra"]
    assert fields["line"] == resolve_off_registry_check().line
