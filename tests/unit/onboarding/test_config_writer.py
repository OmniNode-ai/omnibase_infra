# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for ConfigWriter and write_env_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.onboarding.config_writer import (
    ConfigWriter,
    ConfigWriterError,
    write_env_file,
)

pytestmark = pytest.mark.unit


class TestConfigWriterRender:
    """Tests for the pure render method."""

    def test_render_empty_dict_no_existing(self) -> None:
        writer = ConfigWriter()
        result = writer.render({})
        assert result == ""

    def test_render_basic_pairs(self) -> None:
        writer = ConfigWriter()
        result = writer.render({"FOO": "bar", "BAZ": "qux"})
        assert "FOO=bar" in result
        assert "BAZ=qux" in result

    def test_render_merges_with_existing(self) -> None:
        writer = ConfigWriter()
        existing = "EXISTING_KEY=original_value\nOTHER=kept\n"
        result = writer.render({"NEW_KEY": "new_value"}, existing_content=existing)
        assert "EXISTING_KEY=original_value" in result
        assert "OTHER=kept" in result
        assert "NEW_KEY=new_value" in result

    def test_render_overwrites_existing_key(self) -> None:
        writer = ConfigWriter()
        existing = "KEY=old_value\n"
        result = writer.render({"KEY": "new_value"}, existing_content=existing)
        assert "KEY=new_value" in result
        assert "old_value" not in result

    def test_render_preserves_keys_not_in_dict(self) -> None:
        writer = ConfigWriter()
        existing = "KEEP_ME=yes\nALSO_KEEP=true\n"
        result = writer.render({"NEW": "added"}, existing_content=existing)
        assert "KEEP_ME=yes" in result
        assert "ALSO_KEEP=true" in result
        assert "NEW=added" in result

    def test_render_skips_comments_in_existing(self) -> None:
        writer = ConfigWriter()
        existing = "# This is a comment\nKEY=value\n"
        result = writer.render({"KEY": "value"}, existing_content=existing)
        assert "# This is a comment" not in result
        assert "KEY=value" in result


class TestConfigWriterWrite:
    """Tests for file I/O with atomicity guarantees."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "test.env"
        writer.write({"KEY": "value"}, target)
        assert target.exists()

    def test_write_returns_content_string(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "test.env"
        result = writer.write({"KEY": "value"}, target)
        assert isinstance(result, str)
        assert "KEY=value" in result

    def test_write_file_contains_merged_content(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "test.env"
        writer.write({"KEY": "value"}, target)
        content = target.read_text(encoding="utf-8")
        assert "KEY=value" in content

    def test_write_merge_preserves_existing_keys(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "test.env"
        target.write_text("EXISTING=original\n", encoding="utf-8")
        writer.write({"NEW": "added"}, target)
        content = target.read_text(encoding="utf-8")
        assert "EXISTING=original" in content
        assert "NEW=added" in content

    def test_write_overwrites_existing_key(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "test.env"
        target.write_text("KEY=old\n", encoding="utf-8")
        writer.write({"KEY": "new"}, target)
        content = target.read_text(encoding="utf-8")
        assert "KEY=new" in content
        assert "old" not in content

    def test_write_atomic_no_tmp_file_left_on_success(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "test.env"
        writer.write({"KEY": "value"}, target)
        tmp_files = list(tmp_path.glob(f".{target.name}.tmp.*"))
        assert not tmp_files, f"Leftover tmp files: {tmp_files}"

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "subdir" / "nested" / "test.env"
        writer.write({"KEY": "value"}, target)
        assert target.exists()

    def test_write_empty_dict_with_existing_preserves_all(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "test.env"
        target.write_text("A=1\nB=2\n", encoding="utf-8")
        writer.write({}, target)
        content = target.read_text(encoding="utf-8")
        assert "A=1" in content
        assert "B=2" in content


class TestWriteEnvFile:
    """Tests for the convenience function retained by OMN-10773."""

    def test_write_env_file_uses_config_writer_semantics(self, tmp_path: Path) -> None:
        target = tmp_path / "test.env"
        target.write_text("EXISTING=old\nKEY=old\n", encoding="utf-8")

        content = write_env_file({"KEY": "new"}, target)

        assert "EXISTING=old" in content
        assert "KEY=new" in content
        assert "old\nKEY=old" not in content
        assert target.read_text(encoding="utf-8") == content

    def test_write_env_file_rejects_unsafe_pairs(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigWriterError, match="newline"):
            write_env_file({"KEY": "value\nINJECTED=evil"}, tmp_path / "test.env")


class TestConfigWriterSafety:
    """Safety assertions: tests must never write under real ~/.omnibase/."""

    def test_real_home_omnibase_untouched(self, tmp_path: Path) -> None:
        real_omnibase = Path.home() / ".omnibase"
        writer = ConfigWriter()
        target = tmp_path / "safe.env"
        writer.write({"SAFE": "yes"}, target)

        if real_omnibase.exists():
            assert target.parent == tmp_path
        else:
            assert not real_omnibase.exists()

    def test_tmp_path_is_not_under_home_omnibase(self, tmp_path: Path) -> None:
        real_omnibase = Path.home() / ".omnibase"
        assert not str(tmp_path).startswith(str(real_omnibase))


class TestConfigWriterValidation:
    """Tests for key/value validation in render()."""

    def test_render_rejects_newline_in_key(self) -> None:
        writer = ConfigWriter()
        with pytest.raises(ConfigWriterError, match="newline"):
            writer.render({"KEY\nINJECT": "value"})

    def test_render_rejects_carriage_return_in_key(self) -> None:
        writer = ConfigWriter()
        with pytest.raises(ConfigWriterError, match="carriage return"):
            writer.render({"KEY\rINJECT": "value"})

    def test_render_rejects_equals_sign_in_key(self) -> None:
        writer = ConfigWriter()
        with pytest.raises(ConfigWriterError, match="equals sign"):
            writer.render({"KEY=INJECT": "value"})

    def test_render_rejects_newline_in_value(self) -> None:
        writer = ConfigWriter()
        with pytest.raises(ConfigWriterError, match="newline"):
            writer.render({"KEY": "value\nINJECTED=evil"})

    def test_render_rejects_carriage_return_in_value(self) -> None:
        writer = ConfigWriter()
        with pytest.raises(ConfigWriterError, match="carriage return"):
            writer.render({"KEY": "value\rINJECTED=evil"})
