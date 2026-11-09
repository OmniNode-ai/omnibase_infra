# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for ``--probe-ladder`` (OMN-23164).

Goes through ``click``: the real command, the real option parsing and the real
refusal paths, stopping before dispatch (dispatch needs a co-installed
omnimarket and a live model endpoint, which do not belong in this gate).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration


def _invoke(args: list[str]) -> object:
    return CliRunner().invoke(delegate_command, args, catch_exceptions=False)


class TestProbeLadderOnTheCommandLine:
    def test_a_ladder_without_synthetic_traffic_is_refused_before_dispatch(
        self, tmp_path: Path
    ) -> None:
        ladder = tmp_path / "ladder.yaml"
        ladder.write_text("rungs: []\n", encoding="utf-8")
        state_root = tmp_path / "state"
        result = _invoke(
            [
                "Reply with exactly: OK",
                "--probe-ladder",
                str(ladder),
                "--state-root",
                str(state_root),
            ]
        )
        assert result.exit_code == 2
        assert "--traffic-class synthetic" in result.output
        assert not (state_root / "tmp").exists(), (
            "nothing may be written before refusal"
        )

    def test_a_missing_ladder_file_is_refused_at_the_flag(self, tmp_path: Path) -> None:
        result = _invoke(
            [
                "Reply with exactly: OK",
                "--probe-ladder",
                str(tmp_path / "absent.yaml"),
                "--traffic-class",
                "synthetic",
                "--source-surface",
                "doctor_escalation_check",
                "--state-root",
                str(tmp_path / "state"),
            ]
        )
        assert result.exit_code == 2

    def test_the_flag_is_listed_in_help(self) -> None:
        assert "--probe-ladder" in _invoke(["--help"]).output
