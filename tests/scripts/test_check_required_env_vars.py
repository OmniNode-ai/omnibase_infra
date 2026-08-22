# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for required compose environment validation (OMN-15009, OMN-15537).

OMN-15537: the hook must validate a property of the DIFF (compose file vs. the
checked-in manifest), never a property of the invoking host's environment. Every
test here drives the script with only two committed inputs — a compose fixture and
a manifest fixture — and asserts on exit code alone. None of them touch the process
environment, `~/.omnibase/.env`, or any other host env file; a passing suite here on
one host proves nothing host-specific was exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_required_env_vars


def _write_compose(tmp_path: Path, *var_names: str) -> Path:
    compose = tmp_path / "compose.yml"
    lines = ["services:", "  effects:", "    environment:"]
    for name in var_names:
        lines.append(f"      {name}: ${{{name}:?required}}")
    compose.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return compose


def _write_manifest(tmp_path: Path, *var_names: str) -> Path:
    manifest = tmp_path / "required-env-vars.manifest.txt"
    manifest.write_text(
        "# declared names only\n" + "\n".join(var_names) + ("\n" if var_names else ""),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.unit
def test_matching_compose_and_manifest_passes_with_zero_host_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GREEN: compose and manifest agree — exit 0 with no process env involved."""
    compose = _write_compose(tmp_path, "DEPLOY_AGENT_HMAC_SECRET", "GITHUB_TOKEN")
    manifest = _write_manifest(tmp_path, "DEPLOY_AGENT_HMAC_SECRET", "GITHUB_TOKEN")

    # Prove the check does not consult the process environment: neither var is set,
    # and a prior host-env-reading implementation would fail here.
    monkeypatch.delenv("DEPLOY_AGENT_HMAC_SECRET", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result = check_required_env_vars.main(
        [
            "--compose-file",
            str(compose),
            "--manifest-file",
            str(manifest),
        ]
    )

    assert result == 0


@pytest.mark.unit
def test_new_compose_var_undeclared_in_manifest_fails(tmp_path: Path) -> None:
    """RED: a genuinely new ${VAR:?} added to compose without a manifest entry
    must fail — this is the load-bearing removal/addition-detection vector
    (OMN-15537 AC3), not the vacuous clean-tree-only case."""
    compose = _write_compose(tmp_path, "DEPLOY_AGENT_HMAC_SECRET", "NEW_UNDECLARED_VAR")
    manifest = _write_manifest(tmp_path, "DEPLOY_AGENT_HMAC_SECRET")

    result = check_required_env_vars.main(
        [
            "--compose-file",
            str(compose),
            "--manifest-file",
            str(manifest),
        ]
    )

    assert result == 1


@pytest.mark.unit
def test_manifest_entry_removed_while_compose_still_requires_it_fails(
    tmp_path: Path,
) -> None:
    """RED: deleting a declared name from the manifest while compose still
    requires it must fail (OMN-15537 AC3 — the other half of the removal vector:
    a name genuinely dropped from the manifest, not just an addition to compose)."""
    compose = _write_compose(tmp_path, "DEPLOY_AGENT_HMAC_SECRET", "GITHUB_TOKEN")
    manifest = _write_manifest(
        tmp_path, "DEPLOY_AGENT_HMAC_SECRET"
    )  # GITHUB_TOKEN dropped

    result = check_required_env_vars.main(
        [
            "--compose-file",
            str(compose),
            "--manifest-file",
            str(manifest),
        ]
    )

    assert result == 1


@pytest.mark.unit
def test_stale_manifest_entry_no_longer_in_compose_fails(tmp_path: Path) -> None:
    """A manifest entry for a var no longer referenced by compose is drift too —
    the manifest must track the compose file exactly, not be a superset."""
    compose = _write_compose(tmp_path, "DEPLOY_AGENT_HMAC_SECRET")
    manifest = _write_manifest(
        tmp_path, "DEPLOY_AGENT_HMAC_SECRET", "STALE_REMOVED_VAR"
    )

    result = check_required_env_vars.main(
        [
            "--compose-file",
            str(compose),
            "--manifest-file",
            str(manifest),
        ]
    )

    assert result == 1


@pytest.mark.unit
def test_missing_compose_file_exits_two(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "DEPLOY_AGENT_HMAC_SECRET")

    result = check_required_env_vars.main(
        [
            "--compose-file",
            str(tmp_path / "does-not-exist.yml"),
            "--manifest-file",
            str(manifest),
        ]
    )

    assert result == 2


@pytest.mark.unit
def test_missing_manifest_file_exits_two(tmp_path: Path) -> None:
    compose = _write_compose(tmp_path, "DEPLOY_AGENT_HMAC_SECRET")

    result = check_required_env_vars.main(
        [
            "--compose-file",
            str(compose),
            "--manifest-file",
            str(tmp_path / "does-not-exist.txt"),
        ]
    )

    assert result == 2


@pytest.mark.unit
def test_live_repo_compose_and_manifest_are_in_sync() -> None:
    """Same check the pre-commit hook runs, against the real repo files, so this
    test fails the moment the two committed artifacts drift — without needing a
    running hook or any host-provisioned env."""
    repo_root = Path(__file__).resolve().parents[2]
    compose = repo_root / "docker" / "docker-compose.infra.yml"
    manifest = repo_root / "docker" / "required-env-vars.manifest.txt"

    result = check_required_env_vars.main(
        [
            "--compose-file",
            str(compose),
            "--manifest-file",
            str(manifest),
        ]
    )

    assert result == 0
