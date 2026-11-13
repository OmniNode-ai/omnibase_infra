#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Seam test: the dispatch-venv build marker against the core validator (OMN-19998).

Side A is ``dispatch_venv_promotion.stamp_build_lock``, which writes
``<venv>/.built-from`` (lock_path / lock_sha256 / lock_copy) and a copy of the
uv.lock beside it. Side B is the core validator
``omnibase_core.validation.validator_venv_build_lock.ValidatorVenvBuildLock``,
which reads a ``*.venv-build-lock-facts.json`` document whose surfaces point at
a build lock to judge installed pins against.

This file fails when EITHER side changes shape: the marker reader here uses
literal key names, so a marker key rename, a dropped line, or a copy that does
not match its recorded hash breaks it, and the validator is driven through its
real facts-file contract, so a model or rule rename breaks it too.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from omnibase_core.validation.validator_venv_build_lock import ValidatorVenvBuildLock

pytestmark = pytest.mark.unit

_ENGINE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "omnibase_infra"
    / "handlers"
    / "venv_atomic_install"
    / "dispatch_venv_promotion.py"
)


def _load_engine():
    spec = importlib.util.spec_from_file_location("dispatch_venv_promotion", _ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _load_engine()

# The reader-side names are literals on purpose: this test is the seam, so it
# must not take them from the engine that writes them.
MARKER_NAME = ".built-from"
FACTS_NAME = "run.venv-build-lock-facts.json"


def _make_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    venv.mkdir()
    return venv


def _uv_lock(pins: dict[str, str]) -> str:
    """A minimal uv.lock: one registry-sourced [[package]] block per pin."""
    return "\n".join(
        "[[package]]\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        for name, version in sorted(pins.items())
    )


def read_marker(venv: Path) -> dict[str, str]:
    """Parse the marker's key=value lines the way a reader would."""
    fields: dict[str, str] = {}
    for line in (venv / MARKER_NAME).read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key] = value
    return fields


def build_facts(
    tmp_path: Path,
    venv: Path,
    checkout_lock: Path,
    installed: dict[str, str],
) -> Path:
    """Write a facts document whose surface points at the marker's own copy."""
    marker = read_marker(venv)
    facts = tmp_path / FACTS_NAME
    facts.write_text(
        json.dumps(
            {
                "checkout_lock_path": str(checkout_lock),
                "surfaces": [
                    {
                        "label": "dispatch venv",
                        "build_lock_path": str(venv / marker["lock_copy"]),
                        "installed": installed,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return facts


def _error_codes(issues) -> list[str]:
    return [i.code for i in issues if i.severity.value == "error"]


def _stamp(venv: Path, uv_lock: Path) -> None:
    assert ENGINE.stamp_build_lock(venv, uv_lock) == ENGINE.EXIT_OK


def test_marker_names_a_lock_the_validator_can_judge(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path)
    checkout = tmp_path / "uv.lock"
    checkout.write_text(_uv_lock({"alpha": "1.0.0"}), encoding="utf-8")
    _stamp(venv, checkout)

    facts = build_facts(tmp_path, venv, checkout, {"alpha": "1.0.0"})
    validator = ValidatorVenvBuildLock()
    issues = validator._validate_file(facts, validator._load_contract())

    assert _error_codes(issues) == []


def test_installed_version_not_in_the_build_lock_is_an_error(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path)
    checkout = tmp_path / "uv.lock"
    checkout.write_text(_uv_lock({"alpha": "1.0.0"}), encoding="utf-8")
    _stamp(venv, checkout)

    facts = build_facts(tmp_path, venv, checkout, {"alpha": "2.0.0"})
    validator = ValidatorVenvBuildLock()
    issues = validator._validate_file(facts, validator._load_contract())

    assert _error_codes(issues) == ["surface_registry_pin"]


def test_checkout_lock_moving_after_the_build_is_judged_against_the_copy(
    tmp_path: Path,
) -> None:
    venv = _make_venv(tmp_path)
    checkout = tmp_path / "uv.lock"
    checkout.write_text(_uv_lock({"alpha": "1.0.0"}), encoding="utf-8")
    _stamp(venv, checkout)
    # The checkout moves on after the build; the venv still holds the old copy.
    checkout.write_text(_uv_lock({"alpha": "2.0.0"}), encoding="utf-8")

    facts = build_facts(tmp_path, venv, checkout, {"alpha": "1.0.0"})
    validator = ValidatorVenvBuildLock()
    issues = validator._validate_file(facts, validator._load_contract())

    # judged against the venv's copy, not the moved checkout
    assert _error_codes(issues) == []
    assert "checkout_lock_not_authority" in [i.code for i in issues]


def test_marker_hash_equals_copy_hash(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path)
    checkout = tmp_path / "uv.lock"
    lock_text = _uv_lock({"alpha": "1.0.0"})
    checkout.write_text(lock_text, encoding="utf-8")
    _stamp(venv, checkout)

    marker = read_marker(venv)
    copy = venv / marker["lock_copy"]
    assert copy.read_bytes() == lock_text.encode("utf-8")
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == marker["lock_sha256"]
    assert hashlib.sha256(checkout.read_bytes()).hexdigest() == marker["lock_sha256"]


def test_marker_shape_is_exactly_the_keys_the_reader_needs(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path)
    checkout = tmp_path / "uv.lock"
    checkout.write_text(_uv_lock({"alpha": "1.0.0"}), encoding="utf-8")
    _stamp(venv, checkout)

    assert set(read_marker(venv)) == {"lock_path", "lock_sha256", "lock_copy"}
