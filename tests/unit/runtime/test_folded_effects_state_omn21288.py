# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A role fold keeps existing delegation claims and refuses conflicting state."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from omnibase_infra.runtime import entrypoint_preflight
from omnibase_infra.runtime.entrypoint_preflight import prepare_folded_effects_state

pytestmark = pytest.mark.unit


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ONEX_RUNTIME_FOLDED_PROFILES": "effects",
        "ONEX_STATE_DIR": str(tmp_path / "main"),
        "ONEX_FOLDED_EFFECTS_STATE_ROOT": str(tmp_path / "effects"),
    }


def test_existing_claims_and_future_writes_keep_the_effects_volume(
    tmp_path: Path,
) -> None:
    env = _env(tmp_path)
    source = tmp_path / "effects/delegation"
    source.mkdir(parents=True)
    old = source / "delegation_claims.sqlite"
    with sqlite3.connect(old) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE claims (id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO claims VALUES ('already-completed')")
    main = tmp_path / "main"
    main.mkdir()
    (main / "unrelated-state").write_text("keep")

    prepare_folded_effects_state(env)
    new = main / "delegation/delegation_claims.sqlite"
    assert new.samefile(old)
    with sqlite3.connect(new) as db:
        assert db.execute("SELECT id FROM claims").fetchall() == [
            ("already-completed",)
        ]
        db.execute("INSERT INTO claims VALUES ('new-claim')")
    with sqlite3.connect(old) as db:
        assert db.execute("SELECT count(*) FROM claims").fetchone() == (2,)
    assert (main / "unrelated-state").read_text() == "keep"
    prepare_folded_effects_state(env)  # A restart retains the same directory.
    assert new.samefile(old)


@pytest.mark.parametrize("empty_destination", [False, True])
def test_fresh_volume_or_empty_main_directory_can_boot(
    tmp_path: Path, empty_destination: bool
) -> None:
    if empty_destination:
        (tmp_path / "main/delegation").mkdir(parents=True)
    prepare_folded_effects_state(_env(tmp_path))
    assert (tmp_path / "main/delegation").is_dir()
    assert (tmp_path / "main/delegation").samefile(tmp_path / "effects/delegation")


def test_conflicting_main_claims_are_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "main/delegation"
    target.mkdir(parents=True)
    (target / "delegation_claims.sqlite").write_bytes(b"main-claims")
    with pytest.raises(ValueError, match="conflicting delegation state"):
        prepare_folded_effects_state(_env(tmp_path))
    assert (target / "delegation_claims.sqlite").read_bytes() == b"main-claims"
    assert not (tmp_path / "effects").exists()


def test_wrong_existing_link_is_never_retargeted(tmp_path: Path) -> None:
    target = tmp_path / "main/delegation"
    target.parent.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    target.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="conflicting directory"):
        prepare_folded_effects_state(_env(tmp_path))
    assert target.samefile(other)


@pytest.mark.parametrize("invalid", ["relative", "overlap", "unfolded"])
def test_invalid_state_handover_is_refused(tmp_path: Path, invalid: str) -> None:
    env = _env(tmp_path)
    if invalid == "relative":
        env["ONEX_STATE_DIR"] = "relative"
    elif invalid == "overlap":
        env["ONEX_FOLDED_EFFECTS_STATE_ROOT"] = env["ONEX_STATE_DIR"]
    else:
        env["ONEX_RUNTIME_FOLDED_PROFILES"] = "workers"
    with pytest.raises(ValueError):
        prepare_folded_effects_state(env)


def test_unconfigured_lanes_are_unchanged(tmp_path: Path) -> None:
    prepare_folded_effects_state({"ONEX_STATE_DIR": str(tmp_path / "main")})
    assert not (tmp_path / "main").exists()


def test_preflight_prepares_claim_state_before_handlers_can_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stamp(env: dict[str, str]) -> int:
        assert (tmp_path / "main/delegation").samefile(tmp_path / "effects/delegation")
        return 0

    monkeypatch.setattr(entrypoint_preflight, "stamp_all_fingerprints", stamp)
    monkeypatch.setattr(entrypoint_preflight, "render_bifrost_contract", lambda env: 0)
    monkeypatch.setattr(entrypoint_preflight, "render_resolver_config", lambda env: 0)
    assert entrypoint_preflight.run_preflight(_env(tmp_path)) == 0


def test_conflict_aborts_preflight_before_any_database_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "main/delegation"
    target.mkdir(parents=True)
    (target / "claim").write_text("keep")

    def must_not_stamp(env: dict[str, str]) -> int:
        pytest.fail("a conflicting claim store must abort before the database step")

    monkeypatch.setattr(entrypoint_preflight, "stamp_all_fingerprints", must_not_stamp)
    assert entrypoint_preflight.run_preflight(_env(tmp_path)) == 2
    assert "FOLDED_EFFECTS_STATE_REFUSED" in capsys.readouterr().err


def test_unset_main_state_root_is_named(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.pop("ONEX_STATE_DIR", None)
    env.pop("ONEX_STATE_ROOT", None)
    with pytest.raises(ValueError, match="requires ONEX_STATE_DIR"):
        prepare_folded_effects_state(env)
    assert not (tmp_path / "effects").exists()


def test_state_root_alone_is_not_the_claim_readers_directory(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env["ONEX_STATE_ROOT"] = env.pop("ONEX_STATE_DIR")
    with pytest.raises(ValueError, match="requires ONEX_STATE_DIR"):
        prepare_folded_effects_state(env)
    assert not (tmp_path / "effects").exists()
    assert not (tmp_path / "main").exists()


@pytest.mark.parametrize("file_path", ["effects", "effects/delegation", "main"])
def test_state_path_that_is_a_regular_file_is_named(
    tmp_path: Path, file_path: str
) -> None:
    blocker = tmp_path / file_path
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a directory")
    with pytest.raises(ValueError, match=f"must be a directory: {blocker}"):
        prepare_folded_effects_state(_env(tmp_path))
    assert blocker.read_text() == "not a directory"
    assert not (tmp_path / "main/delegation").exists()


def test_source_swapped_during_handover_refuses_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "effects/delegation"
    outside = tmp_path / "outside"
    outside.mkdir()
    link = Path.symlink_to

    def swap_then_link(
        self: Path, target: Path, target_is_directory: bool = False
    ) -> None:
        source.rmdir()
        link(source, outside, True)
        link(self, target, target_is_directory)

    monkeypatch.setattr(Path, "symlink_to", swap_then_link)
    with pytest.raises(ValueError, match="changed during handover"):
        prepare_folded_effects_state(_env(tmp_path))


@pytest.mark.parametrize("linked_path", ["effects", "effects/delegation", "main"])
def test_symlinked_state_paths_are_refused_without_touching_the_target(
    tmp_path: Path, linked_path: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    claim = outside / "claim"
    claim.write_text("preserve")
    link = tmp_path / linked_path
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        prepare_folded_effects_state(_env(tmp_path))

    assert link.is_symlink()
    assert claim.read_text() == "preserve"
    assert list(outside.iterdir()) == [claim]


def test_existing_alias_does_not_accept_a_redirected_source(tmp_path: Path) -> None:
    env = _env(tmp_path)
    prepare_folded_effects_state(env)
    source = tmp_path / "effects/delegation"
    source.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    source.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        prepare_folded_effects_state(env)

    assert (tmp_path / "main/delegation").is_symlink()
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("remove_root", [False, True])
def test_existing_alias_refuses_a_missing_preserved_store(
    tmp_path: Path, remove_root: bool
) -> None:
    env = _env(tmp_path)
    prepare_folded_effects_state(env)
    source = tmp_path / "effects/delegation"
    source.rmdir()
    if remove_root:
        source.parent.rmdir()

    with pytest.raises(ValueError, match="preserved delegation state is missing"):
        prepare_folded_effects_state(env)

    assert (tmp_path / "main/delegation").is_symlink()
    assert not source.exists()
    assert source.parent.exists() is not remove_root
