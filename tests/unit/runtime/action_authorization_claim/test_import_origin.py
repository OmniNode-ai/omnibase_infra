# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard that focused tests import the candidate, never the base project."""

from __future__ import annotations

from pathlib import Path

import pytest

import omnibase_infra


@pytest.mark.unit
def test_omnibase_infra_import_origin_is_this_candidate() -> None:
    candidate_source = Path(__file__).resolve().parents[4] / "src"
    imported = Path(omnibase_infra.__file__).resolve()

    assert imported.is_relative_to(candidate_source)
