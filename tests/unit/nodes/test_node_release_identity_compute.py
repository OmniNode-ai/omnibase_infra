# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the pure release-identity compute handler (OMN-14471).

Fast, subprocess-free tests of ``HandlerReleaseIdentity.handle`` — the pure core of
the ``node_release_identity_compute`` node. Every decision branch is exercised by
constructing a typed ``ModelReleaseIdentityRequest`` directly and asserting the
decision's exit code, stream, message, and reason code. Byte-for-byte equivalence
with the legacy gate is proven separately by
``tests/scripts/test_check_release_identity.py``.
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_release_identity_compute import (
    HandlerReleaseIdentity,
    ModelReleaseIdentityDecision,
    ModelReleaseIdentityRequest,
)

pytestmark = pytest.mark.unit

# A synthetic path label only (never opened); used to assert config-error messages.
_PYPROJECT_PATH = "/workspace/equiv/pyproject.toml"


def _handle(
    *,
    version: str | None,
    tags: tuple[str, ...] = (),
    changed: tuple[str, ...] | None = (),
) -> ModelReleaseIdentityDecision:
    request = ModelReleaseIdentityRequest(
        pyproject_version_raw=version,
        pyproject_path=_PYPROJECT_PATH,
        published_tags=tags,
        changed_files=changed,
    )
    return HandlerReleaseIdentity().handle(request)


class TestConfigErrors:
    def test_missing_version_is_config_error(self) -> None:
        decision = _handle(version=None, tags=("v1.0.0",))
        assert decision.exit_code == 2
        assert decision.stream == "stderr"
        assert decision.reason_code == "no_pyproject_version"
        assert decision.message == f"ERROR: no project.version in {_PYPROJECT_PATH}"

    def test_empty_version_is_config_error(self) -> None:
        decision = _handle(version="", tags=("v1.0.0",))
        assert decision.exit_code == 2
        assert decision.reason_code == "no_pyproject_version"

    def test_malformed_version_is_config_error(self) -> None:
        decision = _handle(version="not-a-version", tags=("v1.0.0",))
        assert decision.exit_code == 2
        assert decision.stream == "stderr"
        assert decision.reason_code == "malformed_pyproject_version"
        assert decision.message.startswith(
            "ERROR: malformed project.version 'not-a-version': "
        )

    def test_config_error_precedes_tag_and_diff_checks(self) -> None:
        # No tags AND undeterminable changed set, but a bad version still wins.
        decision = _handle(version=None, tags=(), changed=None)
        assert decision.exit_code == 2


class TestExemptAndPassPaths:
    def test_no_published_tag_is_ok(self) -> None:
        decision = _handle(version="1.0.0", tags=(), changed=("src/foo.py",))
        assert decision.exit_code == 0
        assert decision.stream == "stdout"
        assert decision.reason_code == "no_published_tag"
        assert decision.message == (
            "OK: no published tag yet — release-identity bump not required."
        )

    def test_no_packaged_change_is_ok(self) -> None:
        decision = _handle(version="1.0.0", tags=("v2.0.0",), changed=("docs/x.md",))
        assert decision.exit_code == 0
        assert decision.reason_code == "no_packaged_change"
        # Even though the version is BEHIND the tag, a docs-only diff is exempt.
        assert "no packaged src/** change" in decision.message
        assert "pyproject 1.0.0" in decision.message
        assert "latest published 2.0.0" in decision.message

    def test_version_ahead_is_ok(self) -> None:
        decision = _handle(version="2.0.0", tags=("v1.0.0",), changed=("src/foo.py",))
        assert decision.exit_code == 0
        assert decision.reason_code == "version_ahead"
        assert decision.message == (
            "OK: version 2.0.0 is ahead of latest published 1.0.0."
        )


class TestFailPath:
    def test_src_change_not_ahead_fails(self) -> None:
        decision = _handle(version="1.0.0", tags=("v1.0.0",), changed=("src/foo.py",))
        assert decision.exit_code == 1
        assert decision.stream == "stderr"
        assert decision.reason_code == "version_not_ahead"
        assert not decision  # __bool__ is False on a failing gate

    def test_fail_message_has_two_lines_and_bump_suggestion(self) -> None:
        decision = _handle(version="1.2.3", tags=("v1.2.3",), changed=("src/foo.py",))
        lines = decision.message.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith(
            "FAIL: packaged source changed but pyproject version 1.2.3 is NOT ahead"
        )
        assert "e.g. 1.2.4" in lines[1]

    def test_version_behind_with_src_change_fails(self) -> None:
        decision = _handle(version="1.0.0", tags=("v2.0.0",), changed=("src/foo.py",))
        assert decision.exit_code == 1
        assert decision.reason_code == "version_not_ahead"


class TestChangedFileDetection:
    def test_none_changed_files_enforces(self) -> None:
        # Undeterminable change set -> enforce; 1.0.0 not ahead of 1.0.0 -> FAIL.
        decision = _handle(version="1.0.0", tags=("v1.0.0",), changed=None)
        assert decision.exit_code == 1

    def test_none_changed_files_still_passes_when_ahead(self) -> None:
        decision = _handle(version="2.0.0", tags=("v1.0.0",), changed=None)
        assert decision.exit_code == 0
        assert decision.reason_code == "version_ahead"

    def test_empty_changed_files_is_exempt(self) -> None:
        # An explicit empty change set is NOT undeterminable -> no packaged change.
        decision = _handle(version="1.0.0", tags=("v2.0.0",), changed=())
        assert decision.exit_code == 0
        assert decision.reason_code == "no_packaged_change"

    def test_nested_src_path_detected(self) -> None:
        decision = _handle(
            version="1.0.0", tags=("v1.0.0",), changed=("docs/a.md", "src/deep/mod.py")
        )
        assert decision.exit_code == 1


class TestTagSelection:
    def test_bare_semver_tag_without_v_prefix(self) -> None:
        decision = _handle(version="2.0.0", tags=("1.5.0",), changed=("src/foo.py",))
        assert decision.exit_code == 0
        assert "latest published 1.5.0" in decision.message

    def test_highest_tag_selected_and_garbage_skipped(self) -> None:
        decision = _handle(
            version="1.4.0",
            tags=("v1.0.0", "v1.3.0", "garbage", "", "v1.2.0"),
            changed=("src/foo.py",),
        )
        assert decision.exit_code == 0
        assert "latest published 1.3.0" in decision.message

    def test_only_unparseable_tags_treated_as_no_tag(self) -> None:
        decision = _handle(
            version="1.0.0", tags=("garbage", "not-a-tag"), changed=("src/foo.py",)
        )
        assert decision.exit_code == 0
        assert decision.reason_code == "no_published_tag"


class TestHandlerClassification:
    def test_handler_type_and_category(self) -> None:
        handler = HandlerReleaseIdentity()
        assert handler.handler_type == EnumHandlerType.COMPUTE_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.COMPUTE

    def test_handler_is_pure_same_input_same_output(self) -> None:
        request = ModelReleaseIdentityRequest(
            pyproject_version_raw="2.0.0",
            pyproject_path=_PYPROJECT_PATH,
            published_tags=("v1.0.0",),
            changed_files=("src/foo.py",),
        )
        first = HandlerReleaseIdentity().handle(request)
        second = HandlerReleaseIdentity().handle(request)
        assert first == second
