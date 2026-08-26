# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for CredentialsWriter and write_credentials_file (OMN-16035).

The credentials artifact is the JSON sibling of ``ConfigWriter``'s env output.
Its distinguishing invariants — explicit ``chmod 0600``, atomic replace, and
deep merge-and-preserve — are asserted directly here rather than inferred from
"no exception raised".
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from omnibase_core.types import StrictJsonType
from omnibase_infra.onboarding.credentials_writer import (
    CREDENTIALS_DIR_MODE,
    CREDENTIALS_FILE_MODE,
    CredentialsWriter,
    CredentialsWriterError,
    select_credential_entries,
    write_credentials_file,
)

pytestmark = pytest.mark.unit


def _mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class TestCredentialsWriterRender:
    """Pure render: no filesystem contact."""

    def test_render_empty_dict_no_existing(self) -> None:
        assert json.loads(CredentialsWriter().render({})) == {}

    def test_render_emits_json_not_yaml(self) -> None:
        rendered = CredentialsWriter().render({"token": "abc"})
        assert json.loads(rendered) == {"token": "abc"}
        assert rendered.endswith("\n")

    def test_render_supports_nested_dicts(self) -> None:
        payload: dict[str, StrictJsonType] = {
            "credentials": {
                "interactive_onboarding": {"OMNI_API_TOKEN": "t-1"},
            }
        }
        parsed = json.loads(CredentialsWriter().render(payload))
        assert (
            parsed["credentials"]["interactive_onboarding"]["OMNI_API_TOKEN"] == "t-1"
        )

    def test_render_is_deterministic_sorted(self) -> None:
        rendered = CredentialsWriter().render({"b": "2", "a": "1"})
        assert rendered.index('"a"') < rendered.index('"b"')

    def test_render_deep_merges_preserving_siblings(self) -> None:
        existing = json.dumps(
            {"credentials": {"policy_a": {"A_TOKEN": "keep"}}, "unrelated": "kept"}
        )
        parsed = json.loads(
            CredentialsWriter().render(
                {"credentials": {"policy_b": {"B_TOKEN": "new"}}},
                existing_content=existing,
            )
        )
        assert parsed["credentials"]["policy_a"] == {"A_TOKEN": "keep"}
        assert parsed["credentials"]["policy_b"] == {"B_TOKEN": "new"}
        assert parsed["unrelated"] == "kept"

    def test_render_overwrites_same_leaf_key(self) -> None:
        existing = json.dumps({"credentials": {"p": {"TOKEN": "old"}}})
        parsed = json.loads(
            CredentialsWriter().render(
                {"credentials": {"p": {"TOKEN": "new"}}}, existing_content=existing
            )
        )
        assert parsed["credentials"]["p"]["TOKEN"] == "new"

    def test_render_scalar_replaces_existing_subtree(self) -> None:
        existing = json.dumps({"credentials": {"p": {"TOKEN": "old"}}})
        parsed = json.loads(
            CredentialsWriter().render(
                {"credentials": "revoked"}, existing_content=existing
            )
        )
        assert parsed["credentials"] == "revoked"

    def test_render_blank_existing_content_is_empty_document(self) -> None:
        parsed = json.loads(
            CredentialsWriter().render({"a": "1"}, existing_content="  ")
        )
        assert parsed == {"a": "1"}

    def test_render_rejects_malformed_existing_json(self) -> None:
        with pytest.raises(CredentialsWriterError, match="not valid JSON"):
            CredentialsWriter().render({"a": "1"}, existing_content="{nope")

    def test_render_rejects_non_object_existing_json(self) -> None:
        with pytest.raises(CredentialsWriterError, match="JSON object"):
            CredentialsWriter().render({"a": "1"}, existing_content="[1, 2]")

    def test_render_rejects_non_string_key(self) -> None:
        payload = {1: "value"}
        with pytest.raises(CredentialsWriterError, match="non-string key"):
            CredentialsWriter().render(payload)  # type: ignore[arg-type]

    def test_render_rejects_non_string_nested_key(self) -> None:
        payload = {"outer": {2: "value"}}
        with pytest.raises(CredentialsWriterError, match="non-string key"):
            CredentialsWriter().render(payload)  # type: ignore[arg-type]

    def test_render_rejects_unserializable_value(self) -> None:
        payload = {"key": object()}
        with pytest.raises(CredentialsWriterError, match="not JSON-serializable"):
            CredentialsWriter().render(payload)  # type: ignore[arg-type]

    def test_render_rejects_nan(self) -> None:
        with pytest.raises(CredentialsWriterError, match="not JSON-serializable"):
            CredentialsWriter().render({"key": float("nan")})


class TestCredentialsWriterPermissions:
    """The stricter permission invariant that distinguishes this writer."""

    def test_write_sets_mode_0600(self, tmp_path: Path) -> None:
        target = tmp_path / "credentials.json"
        CredentialsWriter().write({"token": "abc"}, target)
        assert _mode_of(target) == 0o600
        assert CREDENTIALS_FILE_MODE == 0o600

    def test_write_tightens_permissions_on_preexisting_loose_file(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "credentials.json"
        target.write_text(json.dumps({"existing": "value"}), encoding="utf-8")
        target.chmod(0o644)

        CredentialsWriter().write({"token": "abc"}, target)

        assert _mode_of(target) == 0o600

    def test_write_creates_parent_dir_with_0700(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dir" / "credentials.json"
        CredentialsWriter().write({"token": "abc"}, target)
        assert target.exists()
        assert _mode_of(target.parent) == CREDENTIALS_DIR_MODE

    def test_write_rejects_target_that_ends_up_world_readable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The postcondition is asserted, not assumed: a defeated chmod must raise."""
        target = tmp_path / "credentials.json"
        real_fchmod = os.fchmod

        def _loose_fchmod(fd: int, mode: int) -> None:
            real_fchmod(fd, 0o644)

        monkeypatch.setattr(os, "fchmod", _loose_fchmod)
        monkeypatch.setattr(os, "chmod", lambda *args, **kwargs: None)

        with pytest.raises(CredentialsWriterError, match="0600"):
            CredentialsWriter().write({"token": "abc"}, target)

        # Fails closed BEFORE the replace: no secret bytes are published.
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []


class TestCredentialsWriterAtomicity:
    """mkstemp + os.replace, never a torn in-place write."""

    def test_write_leaves_no_tmp_file_on_success(self, tmp_path: Path) -> None:
        target = tmp_path / "credentials.json"
        CredentialsWriter().write({"token": "abc"}, target)
        assert list(tmp_path.iterdir()) == [target]

    def test_failed_write_leaves_original_intact(self, tmp_path: Path) -> None:
        target = tmp_path / "credentials.json"
        original = json.dumps({"credentials": {"p": {"TOKEN": "original"}}})
        target.write_text(original, encoding="utf-8")

        with pytest.raises(CredentialsWriterError):
            CredentialsWriter().write({"bad": object()}, target)  # type: ignore[dict-item]

        assert target.read_text(encoding="utf-8") == original

    def test_failed_replace_leaves_no_tmp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "credentials.json"

        def _boom(src: object, dst: object) -> None:
            raise OSError("simulated crash between write and replace")

        monkeypatch.setattr(os, "replace", _boom)

        with pytest.raises(OSError, match="simulated crash"):
            CredentialsWriter().write({"token": "abc"}, target)

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_write_returns_content_matching_file(self, tmp_path: Path) -> None:
        target = tmp_path / "credentials.json"
        content = CredentialsWriter().write({"token": "abc"}, target)
        assert target.read_text(encoding="utf-8") == content


class TestCredentialsWriterMergePreserve:
    """Merge-and-preserve against a real file."""

    def test_write_preserves_unrelated_existing_keys(self, tmp_path: Path) -> None:
        target = tmp_path / "credentials.json"
        target.write_text(
            json.dumps({"credentials": {"policy_a": {"A_TOKEN": "keep"}}}),
            encoding="utf-8",
        )

        CredentialsWriter().write(
            {"credentials": {"policy_b": {"B_TOKEN": "new"}}}, target
        )

        parsed = json.loads(target.read_text(encoding="utf-8"))
        assert parsed["credentials"]["policy_a"] == {"A_TOKEN": "keep"}
        assert parsed["credentials"]["policy_b"] == {"B_TOKEN": "new"}

    def test_write_empty_payload_preserves_everything(self, tmp_path: Path) -> None:
        target = tmp_path / "credentials.json"
        target.write_text(json.dumps({"a": "1", "b": {"c": "2"}}), encoding="utf-8")

        CredentialsWriter().write({}, target)

        assert json.loads(target.read_text(encoding="utf-8")) == {
            "a": "1",
            "b": {"c": "2"},
        }


class TestSelectCredentialEntries:
    """Only credential-shaped env keys reach the 0600 artifact."""

    def test_selects_secret_shaped_keys(self) -> None:
        selected = select_credential_entries(
            {
                "ONEX_API_TOKEN": "t",
                "POSTGRES_PASSWORD": "p",
                "OMNI_CLIENT_SECRET": "s",
                "AWS_ACCESS_KEY_ID": "k",
                "GATEWAY_CREDENTIAL": "c",
            }
        )
        assert set(selected) == {
            "ONEX_API_TOKEN",
            "POSTGRES_PASSWORD",
            "OMNI_CLIENT_SECRET",
            "AWS_ACCESS_KEY_ID",
            "GATEWAY_CREDENTIAL",
        }

    def test_excludes_non_credential_keys(self) -> None:
        assert select_credential_entries({"ONEX_ENV": "dev", "LOG_LEVEL": "INFO"}) == {}

    def test_marker_must_be_a_whole_token(self) -> None:
        """VALKEY_HOST contains "KEY" as a substring but is not a credential."""
        assert (
            select_credential_entries(
                {
                    "VALKEY_HOST": "localhost",
                    "VALKEY_PORT": "16379",
                    "MONKEYPATCH": "no",
                }
            )
            == {}
        )

    def test_empty_mapping(self) -> None:
        assert select_credential_entries({}) == {}


class TestWriteCredentialsFile:
    """Explicit-invocation convenience wrapper."""

    def test_wrapper_matches_writer_semantics(self, tmp_path: Path) -> None:
        target = tmp_path / "credentials.json"
        target.write_text(json.dumps({"keep": "yes"}), encoding="utf-8")

        content = write_credentials_file({"token": "abc"}, target)

        parsed = json.loads(content)
        assert parsed == {"keep": "yes", "token": "abc"}
        assert target.read_text(encoding="utf-8") == content
        assert _mode_of(target) == 0o600


class TestCredentialsWriterSafety:
    """Tests must never write under the real home credentials location."""

    def test_tmp_path_is_not_under_home_onex(self, tmp_path: Path) -> None:
        assert not str(tmp_path).startswith(str(Path.home() / ".onex"))

    def test_no_implicit_write_on_render(self, tmp_path: Path) -> None:
        CredentialsWriter().render({"token": "abc"})
        assert list(tmp_path.iterdir()) == []
