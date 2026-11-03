# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A floor marker cannot turn an unreadable attachment into an authorized one."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.cli import omnimarket_drift_guard as guard
from omnibase_infra.models.delegation.model_floor_marker import ModelFloorMarker


def test_floor_marker_does_not_override_undetermined_attachment(tmp_path: Path) -> None:
    (tmp_path / "omnimarket" / ".git").mkdir(parents=True)
    head = "a" * 40
    (tmp_path / ".onex-workspace-floor.json").write_text(
        json.dumps({"omnimarket_commit": head}), encoding="utf-8"
    )
    with (
        patch.object(guard, "canonical_local_omnimarket_commit", return_value=head),
        patch.object(
            guard,
            "canonical_clone_attachment",
            return_value=guard.CanonicalCloneAttachment.UNDETERMINED,
        ),
        patch.object(guard, "installed_omnimarket_commit", return_value=head),
        pytest.raises(guard.OmnimarketDriftError, match="could not prove"),
    ):
        guard.check_omnimarket_drift(str(tmp_path))


@pytest.mark.parametrize(
    "raw", [b"\xff", b"[]", b"null", b"{", b'{"omnimarket_commit": 1}']
)
def test_unreadable_or_malformed_floor_names_nothing(
    tmp_path: Path, raw: bytes
) -> None:
    (tmp_path / ".onex-workspace-floor.json").write_bytes(raw)
    assert not ModelFloorMarker.names_commit(tmp_path, "a" * 40)
