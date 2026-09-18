# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration proof for ledger-roll-stable ACL apply consent citations."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.validation.application_database_acl_apply import (
    resolve_consent_citation,
)

pytestmark = pytest.mark.integration

_CONSENT_STAMP = "2026-09-14T12:11:31Z"
_CONSENT_ROW = (
    "2026-09-14T12:11:31Z | OPERATOR-CONSENT | lane=omn15355-change-window | "
    'approved_by=operator | "authorized" | APPROVED SCOPE: the OMN-15355 '
    "change window: live GRANT/REVOKE of PostgreSQL role and object privileges "
    "per the generated ACL matrix on the dev lane then onex-dev | OUT OF SCOPE: "
    "onex-prod; the public cluster | This row is the durable authorization "
    "evidence"
)


def _ledger(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "LEDGER.md"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_timestamp_consent_citation_resolves_after_ledger_roll(
    tmp_path: Path,
) -> None:
    """A timestamp citation follows the consent row into the ledger archive."""
    live_ledger = _ledger(
        tmp_path,
        "2026-09-01T00:00:00Z | CLAIM | lane=omn15355-change-window",
    )
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "LEDGER_2026-09-15-split.md").write_text(
        "## Rolled ledger segment\n\n" + _CONSENT_ROW + "\n",
        encoding="utf-8",
    )

    consent = resolve_consent_citation(
        f"{live_ledger}@{_CONSENT_STAMP}",
        ticket="OMN-15355",
        ledger_root=tmp_path,
    )

    assert consent.approved_by == "operator"
    assert consent.lane == "omn15355-change-window"
    assert consent.ledger_path.endswith("LEDGER_2026-09-15-split.md")
    assert consent.line_number == 3


def test_a_symlinked_archive_entry_is_refused_end_to_end(tmp_path: Path) -> None:
    """A planted symlink must not authorise a live privilege change.

    The archive glob restricts the NAME a candidate may have; it does not
    restrict what that name points at, and ``read_text`` follows symlinks. This
    exercises the whole resolution path the ACL apply uses, with a link that
    matches the roll pattern exactly and points at a real consent row outside
    the tree.
    """
    from omnibase_infra.validation.application_database_acl_apply import (
        AclApplyRefusalError,
    )

    ledger = _ledger(tmp_path, "header", "2026-09-01T00:00:00Z | ROW | lane=x")
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    planted = tmp_path / "planted.md"
    planted.write_text(_CONSENT_ROW + "\n", encoding="utf-8")
    (archive_dir / f"{ledger.stem}_2026-09-15-split.md").symlink_to(planted)

    with pytest.raises(AclApplyRefusalError, match="opens with the timestamp"):
        resolve_consent_citation(
            f"{ledger}@{_CONSENT_STAMP}", ticket="OMN-15355", ledger_root=tmp_path
        )


def test_a_symlinked_archive_directory_is_refused_end_to_end(
    tmp_path: Path,
) -> None:
    """The directory can be the link, and an entry-only check passes on it.

    Every entry inside a symlinked directory resolves consistently, so checking
    entries alone reads a tree somewhere else entirely while looking correct.
    """
    from omnibase_infra.validation.application_database_acl_apply import (
        AclApplyRefusalError,
    )

    ledger = _ledger(tmp_path, "header", "2026-09-01T00:00:00Z | ROW | lane=x")
    planted_dir = tmp_path / "planted_archive"
    planted_dir.mkdir()
    (planted_dir / f"{ledger.stem}_2026-09-15-split.md").write_text(
        _CONSENT_ROW + "\n", encoding="utf-8"
    )
    (tmp_path / "archive").symlink_to(planted_dir, target_is_directory=True)

    with pytest.raises(AclApplyRefusalError, match="opens with the timestamp"):
        resolve_consent_citation(
            f"{ledger}@{_CONSENT_STAMP}", ticket="OMN-15355", ledger_root=tmp_path
        )
