# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The tests an ``onex delegate`` code run is graded by (OMN-17427).

The dev lane's gate refused all 125 Qwen code runs as INSUFFICIENT because no
named test run reached it, 84 of them correct answers. omnimarket's delegation
port runs a code answer against the tests the request names, and reads them
from the request ``metadata`` map (``omnimarket.models.delegation.wire.
model_named_test_run_request``):

* ``named_tests``: the pytest node ids to run, separated by whitespace;
* ``named_tests_answer_path``: the workspace-relative file the answer's code is
  written to before the tests run;
* ``named_tests_files``: a JSON object of workspace-relative path to source, so
  the request carries its tests and the runner needs no caller workspace.

This model turns three CLI flags into those keys, reading each test file's
source here. The limits are the consumer's own, so a request it would refuse is
refused at the flag instead. A partial or unsafe naming is a usage error, never
dropped: a caller that believes a code answer was graded by its tests and was
not is the defect being closed. The ``metadata`` map is accepted by every
released request consumer, so no consumer changes shape.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MAX_NAMED_TESTS",
    "MAX_NAMED_TEST_FILES",
    "MAX_NAMED_TEST_FILE_CHARS",
    "NAMED_TESTS_ANSWER_PATH_METADATA_KEY",
    "NAMED_TESTS_FILES_METADATA_KEY",
    "NAMED_TESTS_METADATA_KEY",
    "ModelDelegateNamedTests",
]

NAMED_TESTS_METADATA_KEY = "named_tests"
NAMED_TESTS_ANSWER_PATH_METADATA_KEY = "named_tests_answer_path"
NAMED_TESTS_FILES_METADATA_KEY = "named_tests_files"

#: The consumer's limits (omnimarket ``ModelNamedTestSpec``).
MAX_NAMED_TESTS = 20
MAX_NAMED_TEST_FILES = 20
MAX_NAMED_TEST_FILE_CHARS = 200_000


def _sandbox_path(flag: str, value: str) -> str:
    """A path that stays inside the sandbox the runner builds, as POSIX text."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise ValueError(
            f"{flag} {value!r} must stay inside the workspace: give a relative "
            "path without '..' or '.git'."
        )
    return path.as_posix()


class ModelDelegateNamedTests(BaseModel):
    """The tests a request names, the files that hold them, and the answer file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    test_ids: tuple[str, ...] = Field(..., min_length=1, max_length=MAX_NAMED_TESTS)
    test_files: tuple[tuple[str, str], ...] = Field(
        ..., min_length=1, max_length=MAX_NAMED_TEST_FILES
    )
    answer_path: str = Field(..., min_length=1)

    @classmethod
    def from_flags(
        cls,
        named_tests: tuple[str, ...],
        test_files: tuple[str, ...],
        answer_path: str | None,
    ) -> ModelDelegateNamedTests | None:
        """The tests the flags name, or None when they name none.

        A test file alone names every test in it. Raises ValueError, naming the
        flag at fault, when the naming is partial, unsafe or over a limit.
        """
        if not named_tests and not test_files:
            if answer_path is not None:
                raise ValueError(
                    "--answer-path names where an answer's code is written for "
                    "its tests to run, so it needs --test-file (and optionally "
                    "--named-test)."
                )
            return None
        if answer_path is None:
            raise ValueError(
                "--test-file and --named-test need --answer-path: the file the "
                "answer's code is written to before the tests run."
            )
        if not test_files:
            raise ValueError(
                "--named-test needs --test-file: the request carries its tests, "
                "so a node id must point into a file given with --test-file."
            )
        carried = cls._read_test_files(test_files)
        answer = _sandbox_path("--answer-path", answer_path.strip())
        if answer in dict(carried):
            raise ValueError(
                f"--answer-path {answer!r} would write the answer over a test file."
            )
        ids = tuple(named_tests) or tuple(path for path, _ in carried)
        if len(ids) > MAX_NAMED_TESTS:
            raise ValueError(
                f"--named-test names {len(ids)} tests; a request carries at most "
                f"{MAX_NAMED_TESTS}."
            )
        for test_id in ids:
            cls._check_test_id(test_id, tuple(path for path, _ in carried))
        return cls(test_ids=ids, test_files=carried, answer_path=answer)

    @staticmethod
    def _read_test_files(test_files: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        if len(test_files) > MAX_NAMED_TEST_FILES:
            raise ValueError(
                f"--test-file names {len(test_files)} files; a request carries "
                f"at most {MAX_NAMED_TEST_FILES}."
            )
        carried: dict[str, str] = {}
        for raw in test_files:
            source = Path(raw)
            # An absolute path lands at its own name; a relative one keeps its
            # directory, so a node id spelled from the caller's tree still holds.
            target = _sandbox_path(
                "--test-file", source.name if source.is_absolute() else raw
            )
            try:
                content = source.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"--test-file {raw!r} cannot be read as UTF-8 text: {exc}"
                ) from exc
            if len(content) > MAX_NAMED_TEST_FILE_CHARS:
                raise ValueError(
                    f"--test-file {raw!r} is {len(content)} characters; a request "
                    f"carries at most {MAX_NAMED_TEST_FILE_CHARS} per file."
                )
            if target in carried and carried[target] != content:
                raise ValueError(
                    f"--test-file {raw!r} lands at {target!r}, which another "
                    "--test-file with different content already holds."
                )
            carried[target] = content
        return tuple(carried.items())

    @staticmethod
    def _check_test_id(test_id: str, carried_paths: tuple[str, ...]) -> None:
        if not test_id or test_id.startswith("-"):
            raise ValueError(f"--named-test {test_id!r} is not a pytest node id.")
        if any(ch.isspace() for ch in test_id):
            raise ValueError(
                f"--named-test {test_id!r} contains whitespace: the request "
                "separates node ids with whitespace, so one cannot hold any."
            )
        file_part = _sandbox_path("--named-test", test_id.split("::", 1)[0])
        if file_part not in carried_paths:
            raise ValueError(
                f"--named-test {test_id!r} is not in any --test-file: the request "
                f"carries {', '.join(carried_paths)}."
            )

    def as_metadata(self) -> dict[str, str]:
        """The request metadata keys, as text."""
        return {
            NAMED_TESTS_METADATA_KEY: " ".join(self.test_ids),
            NAMED_TESTS_ANSWER_PATH_METADATA_KEY: self.answer_path,
            NAMED_TESTS_FILES_METADATA_KEY: json.dumps(
                dict(self.test_files), sort_keys=True
            ),
        }
