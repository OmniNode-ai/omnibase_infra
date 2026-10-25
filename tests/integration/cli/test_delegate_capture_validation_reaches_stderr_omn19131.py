# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""No pre-publish validation error stays only in the capture log (OMN-19131 AC6).

AC6 asks that a re-measurement of the capture logs finds zero runs whose
``ValidationError`` never reached stderr. The lab half of that measurement is a
readback; this is the half a test can hold: drive the real ``onex delegate``
command into the pre-publish refusal, parse the ``ValidationError`` the runtime
wrote into the run's capture log, and require every refused location and error
type it names, and the refusing model, on the operator's stderr.

The capture log is read, not assumed: the test parses the runtime's own
pydantic block, so a refusal whose text changes shape is still compared
location by location. The control runs the identical command without
``--timeout`` and its capture log holds no ``ValidationError`` at all, so the
comparison above is about the refused field and nothing else.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.integration.cli.test_delegate_pre_publish_failure_omn19131 import (
    _invoke,
    _sole_capture_log,
    stand_in_contract,
)

pytestmark = pytest.mark.integration

# Imported for its autouse effect: the timeout-less stand-in request model.
__all__ = ["stand_in_contract"]

_HEADER = re.compile(
    r"ValidationError: (?P<count>\d+) validation errors? for (?P<model>\w+)$"
)
_MESSAGE = re.compile(r"^  \S.*\[type=(?P<type>[a-z_]+)[,\]]")


def _validation_errors(capture: str) -> list[tuple[str, str, str]]:
    """Every (model, location, error type) the capture log's tracebacks name."""
    lines = capture.splitlines()
    found: list[tuple[str, str, str]] = []
    for index, line in enumerate(lines):
        header = _HEADER.search(line)
        if header is None:
            continue
        expected = int(header["count"])
        pairs: list[tuple[str, str, str]] = []
        for location, message in zip(
            lines[index + 1 :], lines[index + 2 :], strict=False
        ):
            matched = _MESSAGE.match(message)
            if matched is not None and not location.startswith(" "):
                pairs.append((header["model"], location.strip(), matched["type"]))
                if len(pairs) == expected:
                    break
        assert len(pairs) == expected, (
            f"could not parse {expected} errors after {line!r}: {pairs}"
        )
        found.extend(pairs)
    return found


def test_every_captured_validation_error_reaches_stderr(tmp_path: Path) -> None:
    """AC6: nothing the runtime refused is visible only in the capture log."""
    result, state_root = _invoke(tmp_path, "--timeout", "60")
    capture = _sole_capture_log(state_root).read_text(encoding="utf-8")

    errors = _validation_errors(capture)
    assert errors, "the forced refusal left no ValidationError in the capture log"
    assert result.exit_code != 0
    for model, location, error_type in errors:
        assert location in result.stderr, f"{location} is only in the capture log"
        assert error_type in result.stderr, f"{error_type} is only in the capture log"
        assert model in result.stderr, f"{model} is only in the capture log"


def test_a_run_the_model_accepts_captures_no_validation_error(
    tmp_path: Path,
) -> None:
    """Control: without ``--timeout`` the same command logs no ValidationError."""
    result, state_root = _invoke(tmp_path)
    capture = _sole_capture_log(state_root).read_text(encoding="utf-8")

    assert result.exit_code == 0, result.stderr
    assert "ValidationError" not in capture
    assert _validation_errors(capture) == []
