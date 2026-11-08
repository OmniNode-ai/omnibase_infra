# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for ``--traffic-class`` and ``--source-surface`` (OMN-23163).

The unit module drives the payload writer directly. This one goes through
``click``: the real command, the real option parsing and the real refusal path,
stopping before dispatch (dispatch needs a co-installed omnimarket and a live
model endpoint, which do not belong in this gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration


def _invoke(args: list[str]) -> object:
    return CliRunner().invoke(delegate_command, args, catch_exceptions=False)


class TestTrafficClassFlagsOnTheCommandLine:
    def test_synthetic_without_a_surface_is_refused_before_dispatch(
        self, tmp_path: Path
    ) -> None:
        result = _invoke(
            [
                "Reply with exactly: OK",
                "--traffic-class",
                "synthetic",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code == 2
        assert "source_surface" in result.output
        assert not (tmp_path / "tmp").exists(), "nothing may be written before refusal"

    def test_an_unknown_traffic_class_is_refused_at_the_flag(
        self, tmp_path: Path
    ) -> None:
        result = _invoke(
            [
                "Reply with exactly: OK",
                "--traffic-class",
                "bogus",
                "--source-surface",
                "probe-ladder",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code == 2
        assert "bogus" in result.output

    def test_both_flags_are_listed_in_help(self) -> None:
        result = _invoke(["--help"])
        assert "--traffic-class" in result.output
        assert "--source-surface" in result.output
