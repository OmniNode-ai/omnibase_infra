# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelFileSystemRequest validators.

Covers:
    - validate_path_security: all rejection and acceptance branches
    - validate_content_byte_size: None passthrough, byte-limit enforcement, valid content
    - validate_operation_requirements (model_validator): WRITE/READ/DELETE/LIST/MKDIR
"""

from __future__ import annotations

import string
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.filesystem.enum_file_system_operation import (
    EnumFileSystemOperation,
)
from omnibase_infra.handlers.filesystem.model_file_system_request import (
    ModelFileSystemRequest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**kwargs: object) -> ModelFileSystemRequest:
    """Build a minimal valid READ request, overriding fields via kwargs."""
    defaults: dict[str, object] = {
        "operation": EnumFileSystemOperation.READ,
        "path": "data/file.txt",
        "correlation_id": uuid4(),
    }
    defaults.update(kwargs)
    return ModelFileSystemRequest(**defaults)  # type: ignore[arg-type]


def _make_write(**kwargs: object) -> ModelFileSystemRequest:
    """Build a minimal valid WRITE request."""
    defaults: dict[str, object] = {
        "operation": EnumFileSystemOperation.WRITE,
        "path": "data/out.txt",
        "content": "hello",
        "correlation_id": uuid4(),
    }
    defaults.update(kwargs)
    return ModelFileSystemRequest(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_path_security — rejection cases
# ---------------------------------------------------------------------------


class TestValidatePathSecurityRejects:
    def test_empty_path_raises(self) -> None:
        with pytest.raises(ValidationError, match="empty or whitespace"):
            _make_request(path="")

    def test_whitespace_only_path_raises(self) -> None:
        with pytest.raises(ValidationError, match="empty or whitespace"):
            _make_request(path="   ")

    def test_dotdot_traversal_raises(self) -> None:
        with pytest.raises(ValidationError, match="parent directory traversal"):
            _make_request(path="../etc/passwd")

    def test_dotdot_mid_path_raises(self) -> None:
        with pytest.raises(ValidationError, match="parent directory traversal"):
            _make_request(path="foo/../bar")

    def test_unix_absolute_path_raises(self) -> None:
        with pytest.raises(ValidationError, match="Absolute paths are not allowed"):
            _make_request(path="/etc/passwd")

    def test_windows_drive_letter_raises(self) -> None:
        with pytest.raises(ValidationError, match="Absolute paths are not allowed"):
            _make_request(path="C:\\Windows\\system32")

    def test_windows_drive_letter_forward_slash_raises(self) -> None:
        with pytest.raises(ValidationError, match="Absolute paths are not allowed"):
            _make_request(path="C:/Windows/system32")

    def test_null_byte_raises(self) -> None:
        with pytest.raises(ValidationError, match="null bytes"):
            _make_request(path="data\x00file.txt")

    def test_control_character_tab_raises(self) -> None:
        with pytest.raises(ValidationError, match="control character"):
            _make_request(path="data\tfile.txt")

    def test_control_character_newline_raises(self) -> None:
        with pytest.raises(ValidationError, match="control character"):
            _make_request(path="data\nfile.txt")

    def test_control_character_carriage_return_raises(self) -> None:
        with pytest.raises(ValidationError, match="control character"):
            _make_request(path="data\rfile.txt")

    def test_control_character_ord_1_raises(self) -> None:
        # ord=1 < 32 — generic low-control char
        with pytest.raises(ValidationError, match="control character"):
            _make_request(path="data\x01file.txt")

    def test_path_exceeds_4096_chars_raises(self) -> None:
        long_path = "a/" * 2049  # > 4096 chars
        with pytest.raises(ValidationError, match="maximum length of 4096"):
            _make_request(path=long_path)

    def test_filename_segment_exceeds_255_chars_raises(self) -> None:
        long_name = "a" * 256
        with pytest.raises(ValidationError, match="maximum length of 255"):
            _make_request(path=f"data/{long_name}")

    # Windows reserved device names
    @pytest.mark.parametrize(
        "reserved",
        [
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        ],
    )
    def test_reserved_windows_name_raises(self, reserved: str) -> None:
        with pytest.raises(ValidationError, match="reserved Windows device name"):
            _make_request(path=f"data/{reserved}")

    def test_reserved_windows_name_with_trailing_dot_raises(self) -> None:
        with pytest.raises(ValidationError, match="reserved Windows device name"):
            _make_request(path="data/CON.")

    def test_reserved_windows_name_with_trailing_space_raises(self) -> None:
        # CON followed by space — Windows still treats it as CON
        with pytest.raises(ValidationError, match="reserved Windows device name"):
            _make_request(path="data/CON ")

    def test_reserved_windows_name_lowercase_raises(self) -> None:
        with pytest.raises(ValidationError, match="reserved Windows device name"):
            _make_request(path="data/con")

    def test_reserved_windows_name_with_extension_raises(self) -> None:
        with pytest.raises(ValidationError, match="reserved Windows device name"):
            _make_request(path="data/NUL.txt")


# ---------------------------------------------------------------------------
# validate_path_security — acceptance cases
# ---------------------------------------------------------------------------


class TestValidatePathSecurityAccepts:
    def test_simple_relative_path(self) -> None:
        req = _make_request(path="data/file.txt")
        assert req.path == "data/file.txt"

    def test_single_segment_filename(self) -> None:
        req = _make_request(path="file.txt")
        assert req.path == "file.txt"

    def test_nested_relative_path(self) -> None:
        req = _make_request(path="a/b/c/d.json")
        assert req.path == "a/b/c/d.json"

    def test_filename_exactly_255_chars(self) -> None:
        name = "a" * 255
        req = _make_request(path=f"dir/{name}")
        assert req.path == f"dir/{name}"

    def test_path_exactly_4096_chars(self) -> None:
        # Build a path of exactly 4096 chars using short repeating segments.
        # "a/" is 2 chars; 2047 repetitions = 4094 chars; + "ab" = 4096.
        # Last segment "ab" is 2 chars (well under 255-char filename limit).
        path = "a/" * 2047 + "ab"
        assert len(path) == 4096
        req = _make_request(path=path)
        assert req.path == path

    def test_path_with_dots_in_name_not_traversal(self) -> None:
        # ".hidden" is fine; only ".." triggers traversal check
        req = _make_request(path=".hidden/config")
        assert req.path == ".hidden/config"

    def test_path_with_numbers_and_hyphens(self) -> None:
        req = _make_request(path="2026-06-28/report-final.csv")
        assert req.path == "2026-06-28/report-final.csv"

    def test_path_with_printable_ascii(self) -> None:
        # All printable ASCII chars with ord >= 32 are permitted in names
        safe_name = string.ascii_letters + string.digits + "-_."
        req = _make_request(path=f"dir/{safe_name}")
        assert req.path == f"dir/{safe_name}"


# ---------------------------------------------------------------------------
# validate_content_byte_size
# ---------------------------------------------------------------------------


class TestValidateContentByteSize:
    def test_none_content_passes(self) -> None:
        req = _make_request(path="data/file.txt", content=None)
        assert req.content is None

    def test_valid_ascii_content_passes(self) -> None:
        req = _make_write(content="hello world")
        assert req.content == "hello world"

    def test_valid_multibyte_utf8_content_passes(self) -> None:
        # A few emoji — well under 10MB
        content = "🎉" * 100
        req = _make_write(content=content)
        assert req.content == content

    def test_content_exactly_at_10mb_byte_limit_passes(self) -> None:
        # 10 * 1024 * 1024 = 10485760 bytes; ASCII chars are 1 byte each
        content = "x" * 10485760
        req = _make_write(content=content)
        assert len(req.content.encode("utf-8")) == 10485760  # type: ignore[union-attr]

    def test_multibyte_content_exceeding_10mb_bytes_raises(self) -> None:
        # Each emoji is 4 bytes in UTF-8; 2_621_441 * 4 = 10_485_764 bytes > 10MB
        # But pydantic max_length is 10_485_760 chars, so we need fewer chars
        # that encode to more than 10MB bytes.
        # Use a 2-byte char (e.g. U+00E9 é = 2 bytes in UTF-8).
        # 5_242_881 * 2 = 10_485_762 bytes > 10_485_760 limit
        # and 5_242_881 chars < 10_485_760 char limit — so pydantic max_length passes
        # but byte validator fires.
        content = "é" * 5_242_881  # é = 2 bytes; total ~10.00MB+2 bytes
        with pytest.raises(ValidationError, match="10MB byte limit"):
            _make_write(content=content)

    def test_empty_string_content_passes(self) -> None:
        req = _make_write(content="")
        assert req.content == ""


# ---------------------------------------------------------------------------
# validate_operation_requirements (model_validator)
# ---------------------------------------------------------------------------


class TestValidateOperationRequirements:
    # WRITE
    def test_write_with_content_passes(self) -> None:
        req = _make_write(content="data here")
        assert req.operation == EnumFileSystemOperation.WRITE
        assert req.content == "data here"

    def test_write_without_content_raises(self) -> None:
        with pytest.raises(ValidationError, match="WRITE operation requires content"):
            ModelFileSystemRequest(
                operation=EnumFileSystemOperation.WRITE,
                path="output/file.txt",
                content=None,
                correlation_id=uuid4(),
            )

    # READ
    def test_read_without_content_passes(self) -> None:
        req = _make_request(operation=EnumFileSystemOperation.READ, path="src/file.py")
        assert req.content is None

    def test_read_with_content_raises(self) -> None:
        with pytest.raises(
            ValidationError, match="READ operation should not have content"
        ):
            ModelFileSystemRequest(
                operation=EnumFileSystemOperation.READ,
                path="src/file.py",
                content="some content",
                correlation_id=uuid4(),
            )

    # DELETE
    def test_delete_without_content_passes(self) -> None:
        req = _make_request(
            operation=EnumFileSystemOperation.DELETE, path="tmp/old.log"
        )
        assert req.content is None

    def test_delete_with_content_raises(self) -> None:
        with pytest.raises(
            ValidationError, match="DELETE operation should not have content"
        ):
            ModelFileSystemRequest(
                operation=EnumFileSystemOperation.DELETE,
                path="tmp/old.log",
                content="oops",
                correlation_id=uuid4(),
            )

    # LIST
    def test_list_without_content_passes(self) -> None:
        req = _make_request(operation=EnumFileSystemOperation.LIST, path="src/handlers")
        assert req.content is None

    def test_list_with_content_raises(self) -> None:
        with pytest.raises(
            ValidationError, match="LIST operation should not have content"
        ):
            ModelFileSystemRequest(
                operation=EnumFileSystemOperation.LIST,
                path="src/handlers",
                content="unexpected",
                correlation_id=uuid4(),
            )

    # MKDIR
    def test_mkdir_without_content_passes(self) -> None:
        req = _make_request(
            operation=EnumFileSystemOperation.MKDIR, path="output/new_dir"
        )
        assert req.content is None

    def test_mkdir_with_content_raises(self) -> None:
        with pytest.raises(
            ValidationError, match="MKDIR operation should not have content"
        ):
            ModelFileSystemRequest(
                operation=EnumFileSystemOperation.MKDIR,
                path="output/new_dir",
                content="unexpected",
                correlation_id=uuid4(),
            )


# ---------------------------------------------------------------------------
# Model immutability and structural sanity
# ---------------------------------------------------------------------------


class TestModelStructure:
    def test_frozen_model_rejects_mutation(self) -> None:
        req = _make_request(path="data/file.txt")
        with pytest.raises((TypeError, ValidationError)):
            req.path = "other/path"  # type: ignore[misc]

    def test_correlation_id_auto_generated(self) -> None:
        req = ModelFileSystemRequest(
            operation=EnumFileSystemOperation.READ,
            path="data/file.txt",
        )
        assert req.correlation_id is not None

    def test_recursive_defaults_to_none(self) -> None:
        req = _make_request(path="data/file.txt")
        assert req.recursive is None

    def test_recursive_can_be_set(self) -> None:
        req = _make_request(path="data/dir", recursive=True)
        assert req.recursive is True

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ModelFileSystemRequest(  # type: ignore[call-arg]
                operation=EnumFileSystemOperation.READ,
                path="data/file.txt",
                unknown_field="value",
            )
