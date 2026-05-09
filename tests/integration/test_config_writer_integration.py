# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for ConfigWriter filesystem behavior (OMN-10783).

Verifies atomic write semantics and multi-step merge cycles against a real
filesystem (tmp_path). No external services required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.onboarding.config_writer import (
    ConfigWriter,
    ConfigWriterError,
    write_env_file,
)

pytestmark = pytest.mark.integration


class TestConfigWriterIntegration:
    """Integration: ConfigWriter filesystem round-trips."""

    def test_sequential_writes_accumulate_keys(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "omnibase.env"

        writer.write({"A": "1"}, target)
        writer.write({"B": "2"}, target)
        writer.write({"C": "3"}, target)

        content = target.read_text(encoding="utf-8")
        assert "A=1" in content
        assert "B=2" in content
        assert "C=3" in content

    def test_overwrite_cycle_removes_old_value(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "omnibase.env"

        writer.write({"KEY": "original"}, target)
        writer.write({"KEY": "updated"}, target)

        content = target.read_text(encoding="utf-8")
        assert "KEY=updated" in content
        assert "original" not in content

    def test_no_tmp_files_after_repeated_writes(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "omnibase.env"

        for i in range(5):
            writer.write({f"KEY_{i}": str(i)}, target)

        leftover = list(tmp_path.glob(f".{target.name}.tmp.*"))
        assert not leftover, f"Tmp files leaked: {leftover}"

    def test_file_content_matches_returned_string(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "omnibase.env"

        returned = writer.write({"X": "y", "Z": "w"}, target)
        on_disk = target.read_text(encoding="utf-8")

        assert returned == on_disk

    def test_render_dry_run_does_not_create_file(self, tmp_path: Path) -> None:
        writer = ConfigWriter()
        target = tmp_path / "should_not_exist.env"

        writer.render({"KEY": "value"})

        assert not target.exists()

    def test_rejected_pair_does_not_corrupt_existing_file(
        self,
        tmp_path: Path,
    ) -> None:
        writer = ConfigWriter()
        target = tmp_path / "omnibase.env"
        original = "EXISTING=stable\n"
        target.write_text(original, encoding="utf-8")

        with pytest.raises(ConfigWriterError):
            writer.write({"BAD=KEY": "value\nINJECTED=evil"}, target)

        assert target.read_text(encoding="utf-8") == original

    def test_write_env_file_wrapper_merges_on_real_filesystem(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "omnibase.env"
        target.write_text("KEEP=existing\nMODE=old\n", encoding="utf-8")

        returned = write_env_file({"MODE": "interactive"}, target)

        assert returned == target.read_text(encoding="utf-8")
        assert "KEEP=existing" in returned
        assert "MODE=interactive" in returned
        assert "MODE=old" not in returned
