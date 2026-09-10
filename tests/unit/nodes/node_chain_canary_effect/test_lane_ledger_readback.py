# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The consumer half of the link-5 lane declaration (OMN-16964).

The declaration lives in omnimarket (`config/ci_bus_lanes.yaml`); the refusals
live here, with the consumer. Every one of these is a case where a plausible
misconfiguration would otherwise produce a canary that reads the wrong thing
and reports it clean.

No network, no database, no bus — the overlay is a file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.lane_transport import (
    LEDGER_READBACK_DSN_ENV_NAME_VAR,
    ledger_readback_env,
    load_lane_ledger_readback,
)


def _overlay(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.unit
def test_a_declared_dev_lane_resolves_to_the_name(tmp_path: Path) -> None:
    overlay = _overlay(
        tmp_path,
        "lanes:\n  dev:\n    ledger_readback:\n      dsn_env: SOME_DECLARED_NAME\n",
    )

    declaration = load_lane_ledger_readback(overlay, "dev")

    assert declaration.lane == "dev"
    assert declaration.dsn_env == "SOME_DECLARED_NAME"
    assert ledger_readback_env(declaration) == {
        LEDGER_READBACK_DSN_ENV_NAME_VAR: "SOME_DECLARED_NAME"
    }


@pytest.mark.unit
def test_the_exported_name_var_is_distinct_from_the_projection_one() -> None:
    """Two legs, two exported variables — never one serving both.

    If these collided, resolving one leg would silently overwrite the other's
    declaration and the canary would read one relation for both links.
    """
    from omnibase_infra.nodes.node_chain_canary_effect.lane_transport import (
        PROJECTION_READBACK_DSN_ENV_NAME_VAR,
    )

    assert LEDGER_READBACK_DSN_ENV_NAME_VAR != PROJECTION_READBACK_DSN_ENV_NAME_VAR


@pytest.mark.unit
@pytest.mark.parametrize("lane", ["stability", "judge", "prod", "lakshman"])
def test_no_lane_but_dev_may_declare_a_ledger_readback(
    tmp_path: Path, lane: str
) -> None:
    """The canary is a dev-lane instrument; the rest are surfaces it may not read."""
    overlay = _overlay(
        tmp_path,
        f"lanes:\n  {lane}:\n    ledger_readback:\n      dsn_env: SOME_NAME\n",
    )

    with pytest.raises(ValueError, match="may not declare"):
        load_lane_ledger_readback(overlay, lane)


@pytest.mark.unit
def test_a_missing_block_is_refused_rather_than_defaulted(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path, "lanes:\n  dev:\n    broker: somewhere:9092\n")

    with pytest.raises(ValueError, match="ledger_readback"):
        load_lane_ledger_readback(overlay, "dev")


@pytest.mark.unit
def test_a_dsn_declared_where_a_name_belongs_is_refused_without_echoing_it(
    tmp_path: Path,
) -> None:
    """A credential pasted into committed config is a red gate, not a secret."""
    overlay = _overlay(
        tmp_path,
        "lanes:\n  dev:\n    ledger_readback:\n"
        "      dsn_env: postgresql://someone:somevalue@example.invalid:5432/somedb\n",
    )

    with pytest.raises(ValueError) as caught:
        load_lane_ledger_readback(overlay, "dev")

    assert "somevalue" not in str(caught.value)


@pytest.mark.unit
def test_a_name_the_shell_cannot_export_is_refused(tmp_path: Path) -> None:
    """A readback that silently never runs is worse than one that fails loudly."""
    overlay = _overlay(
        tmp_path,
        "lanes:\n  dev:\n    ledger_readback:\n      dsn_env: not-a-posix-name\n",
    )

    with pytest.raises(ValueError, match="POSIX"):
        load_lane_ledger_readback(overlay, "dev")


@pytest.mark.unit
def test_an_unreadable_overlay_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_lane_ledger_readback(tmp_path / "nothing-here.yaml", "dev")


@pytest.mark.unit
def test_the_repo_declaration_resolves_when_the_overlay_is_present() -> None:
    """Guard against the two halves drifting apart in the same repo pair.

    Skips rather than fails when the sibling clone is absent: this file must
    stay hermetic, and the cross-repo agreement is asserted in CI where both
    checkouts exist.
    """
    overlay = (
        Path(__file__).resolve().parents[4]
        / ".ci-bus-overlay"
        / "config"
        / "ci_bus_lanes.yaml"
    )
    if not overlay.is_file():
        pytest.skip("the omnimarket lane overlay is not checked out here")

    declaration = load_lane_ledger_readback(overlay, "dev")
    assert declaration.dsn_env
