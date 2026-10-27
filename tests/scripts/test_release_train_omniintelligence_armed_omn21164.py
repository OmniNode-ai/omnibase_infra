# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The train cuts omniintelligence under its distribution name (OMN-21164).

The row named the package ``omniintelligence``, but the repository's own
``uv.lock`` entry is ``omninode-intelligence``, so the first cut would have
refused at the lock bump. The fixture lock below carries the entry's shape as
the repository ships it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_POLICY = _ROOT / "config" / "release_train_policy.yaml"
_SCRIPT = _ROOT / "scripts" / "ci" / "release_train.py"

_LOCK = """version = 1

[[package]]
name = "omniintelligence"
version = "0.5.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "omninode-intelligence"
version = "0.25.0"
source = { editable = "." }
"""


def _load_release_train():
    spec = importlib.util.spec_from_file_location("release_train_omn21164", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rt = _load_release_train()


def test_omniintelligence_row_is_armed_to_cut() -> None:
    policy = rt.load_policy(_POLICY)["omniintelligence"]
    assert policy.mode is rt.EnumTrainMode.CUT


def test_omniintelligence_row_names_the_distribution() -> None:
    policy = rt.load_policy(_POLICY)["omniintelligence"]
    assert policy.package == "omninode-intelligence"


def test_the_row_package_moves_the_checkout_lock_entry() -> None:
    policy = rt.load_policy(_POLICY)["omniintelligence"]
    bumped = rt.rewrite_lock_root_version(_LOCK, policy.package, "0.25.1")
    assert 'name = "omninode-intelligence"\nversion = "0.25.1"' in bumped
    assert 'name = "omniintelligence"\nversion = "0.5.0"' in bumped


def test_the_old_package_name_is_refused_at_the_lock_bump() -> None:
    """Positive control: the name the row carried cannot bump this lock."""
    with pytest.raises(rt.ReleaseTrainConfigError):
        rt.rewrite_lock_root_version(_LOCK, "omniintelligence", "0.25.1")
