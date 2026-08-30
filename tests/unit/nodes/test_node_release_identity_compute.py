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
    shallow: bool = False,
    from_bundle: bool = False,
) -> ModelReleaseIdentityDecision:
    request = ModelReleaseIdentityRequest(
        pyproject_version_raw=version,
        pyproject_path=_PYPROJECT_PATH,
        published_tags=tags,
        changed_files=changed,
        repo_is_shallow=shallow,
        repo_origin_is_bundle=from_bundle,
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


class TestTagStateFailsClosed:
    """An EMPTY tag set is only a pass when it is CREDIBLE (OMN-17240).

    The gate derives "the latest published version" from ``git tag --list``. Before
    OMN-17240 an empty list unconditionally meant "no published tag yet -> nothing
    to enforce", so ANY transport that lost the tags made the gate silently pass.
    That is exactly what happened: the pre-push remote leg shipped the tree as a
    ``git bundle create ... HEAD`` bundle, which carries no ``refs/tags/`` at all,
    so every lab host evaluated release identity against zero tags.

    The markers are read from the repository's own git state by the collector --
    never from a caller-written file or environment variable, which is the kind of
    self-asserted truth this gate exists to refuse.
    """

    def test_bundle_origin_with_no_tags_fails_closed(self) -> None:
        decision = _handle(version="1.0.0", tags=(), from_bundle=True)
        assert decision.exit_code == 2
        assert decision.stream == "stderr"
        assert decision.reason_code == "tag_state_unavailable"
        assert "OMN-17240" in decision.message
        assert "bundle" in decision.message

    def test_shallow_repo_with_no_tags_fails_closed(self) -> None:
        decision = _handle(version="1.0.0", tags=(), shallow=True)
        assert decision.exit_code == 2
        assert decision.reason_code == "tag_state_unavailable"
        assert "OMN-17240" in decision.message
        assert "shallow" in decision.message

    def test_patch_version_with_no_tags_fails_closed(self) -> None:
        """0.38.15 cannot credibly be the first thing this project ever shipped."""
        decision = _handle(version="0.38.15", tags=())
        assert decision.exit_code == 2
        assert decision.reason_code == "tag_state_unavailable"
        assert "OMN-17240" in decision.message
        assert "0.38.15" in decision.message

    def test_unparseable_tags_plus_bundle_origin_fails_closed(self) -> None:
        """No PARSEABLE published version is the same blind spot as no tags."""
        decision = _handle(version="1.0.0", tags=("garbage",), from_bundle=True)
        assert decision.exit_code == 2
        assert decision.reason_code == "tag_state_unavailable"

    def test_credible_empty_tag_set_still_passes(self) -> None:
        """A pre-first-release repo with a complete, non-bundle checkout is fine."""
        decision = _handle(version="1.0.0", tags=(), changed=("src/foo.py",))
        assert decision.exit_code == 0
        assert decision.reason_code == "no_published_tag"

    def test_markers_are_ignored_once_the_tag_state_is_present(self) -> None:
        """The markers only ever qualify an EMPTY tag set; they never veto a real one."""
        decision = _handle(
            version="2.0.0",
            tags=("v1.0.0",),
            changed=("src/foo.py",),
            shallow=True,
            from_bundle=True,
        )
        assert decision.exit_code == 0
        assert decision.reason_code == "version_ahead"

    def test_config_error_still_precedes_the_tag_state_check(self) -> None:
        decision = _handle(version=None, tags=(), from_bundle=True)
        assert decision.exit_code == 2
        assert decision.reason_code == "no_pyproject_version"


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
