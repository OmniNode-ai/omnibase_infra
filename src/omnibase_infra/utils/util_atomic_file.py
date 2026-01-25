# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Atomic file write utilities.

This module provides primitives for atomic file writes using the temp-file-rename
pattern. Atomic writes ensure that file contents are either completely written
or not written at all - there is no intermediate state where the file contains
partial data.

POSIX Atomicity Guarantees:
    On POSIX systems, rename() is atomic within the same filesystem. This module
    creates the temporary file in the same directory as the target file to ensure
    the rename operation is atomic.

    **IMPORTANT**: The temp file MUST be created in the same directory as the
    target file (using `dir=path.parent`). If the temp file is on a different
    filesystem, the rename becomes a copy-and-delete operation, losing atomicity.

NFS Caveat:
    NFS provides weaker atomicity guarantees. While rename() is still atomic on
    NFSv4+, there may be brief windows where both files are visible to other
    clients. For applications requiring strict consistency on NFS, additional
    locking mechanisms may be needed.

Windows Notes:
    - os.replace() is atomic on Windows since Python 3.3
    - Path.rename() on Windows will fail if the target exists; use os.replace()
    - This module uses os.replace() for cross-platform atomic rename

Durability Note:
    This module provides atomicity (all-or-nothing) but not durability guarantees.
    For systems requiring crash-recovery durability, consider adding fsync() after
    write and before rename. This comes at a performance cost.

Example:
    >>> from pathlib import Path
    >>> from omnibase_infra.utils import write_atomic_bytes
    >>>
    >>> # Write data atomically
    >>> data = b"Hello, World!"
    >>> bytes_written = write_atomic_bytes(Path("/tmp/myfile.txt"), data)
    >>> bytes_written
    13
    >>>
    >>> # Async version (uses asyncio.to_thread internally)
    >>> import asyncio
    >>> from omnibase_infra.utils import write_atomic_bytes_async
    >>> bytes_written = asyncio.run(write_atomic_bytes_async(Path("/tmp/myfile.txt"), data))
    >>> bytes_written
    13

.. versionadded:: 0.10.0
    Created as part of OMN-1524 atomic write utilities.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from uuid import UUID

logger = logging.getLogger(__name__)


def write_atomic_bytes(
    path: Path,
    data: bytes,
    *,
    temp_prefix: str = "",
    temp_suffix: str = ".tmp",
    correlation_id: UUID | None = None,
) -> int:
    """Write bytes to a file atomically using temp-file-rename pattern.

    This function provides atomic file writes by:
    1. Creating a temporary file in the same directory as the target
    2. Writing all data to the temporary file
    3. Atomically renaming the temporary file to the target path

    The rename operation is atomic on POSIX systems when both files are on the
    same filesystem. On Windows, os.replace() provides atomic semantics since
    Python 3.3.

    Args:
        path: Target file path. Parent directory must exist.
        data: Bytes to write to the file.
        temp_prefix: Optional prefix for the temporary file name. Useful for
            debugging to identify the source of temp files.
        temp_suffix: Suffix for the temporary file name. Defaults to ".tmp".
        correlation_id: Optional correlation ID for ONEX logging. When provided,
            errors are logged with correlation context before being raised.

    Returns:
        Number of bytes written to the file.

    Raises:
        InfraConnectionError: If the file cannot be written (permissions, disk full,
            etc.). The underlying OSError is chained via ``from e``. The temporary
            file is cleaned up before raising.

    Example:
        >>> from pathlib import Path
        >>> from omnibase_infra.utils.util_atomic_file import write_atomic_bytes
        >>>
        >>> # Basic atomic write
        >>> path = Path("/tmp/test_atomic.txt")
        >>> bytes_written = write_atomic_bytes(path, b"test data")
        >>> bytes_written
        9
        >>>
        >>> # With debugging prefix
        >>> bytes_written = write_atomic_bytes(
        ...     path,
        ...     b"test data",
        ...     temp_prefix="manifest_",
        ...     correlation_id=UUID("12345678-1234-5678-1234-567812345678"),
        ... )

    Warning:
        The parent directory of ``path`` must exist. This function does not
        create parent directories.

    Warning:
        On NFS, atomicity guarantees are weaker. See module docstring for details.

    Related:
        - handler_manifest_persistence.py: Original pattern implementation
    """
    temp_fd: int | None = None
    temp_path: str | None = None

    try:
        # Create temp file in same directory for atomic rename guarantee
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=temp_suffix,
            prefix=temp_prefix,
            dir=path.parent,
        )

        # Write data to temp file
        with os.fdopen(temp_fd, "wb") as f:
            temp_fd = None  # fdopen takes ownership of fd
            bytes_written = f.write(data)

        # Atomic rename (Path.replace is atomic on both POSIX and Windows 3.3+)
        Path(temp_path).replace(path)
        temp_path = None  # Rename succeeded, no cleanup needed

        return bytes_written

    except OSError as e:
        # Log with correlation context if provided
        if correlation_id is not None:
            logger.exception(
                "Atomic write failed for '%s'",
                path,
                extra={
                    "correlation_id": str(correlation_id),
                    "target_path": str(path),
                    "temp_prefix": temp_prefix,
                    "temp_suffix": temp_suffix,
                    "error_type": type(e).__name__,
                },
            )

        # Cleanup: close fd if fdopen didn't take ownership
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass  # Best effort cleanup

        # Cleanup: remove temp file if it exists
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass  # Best effort cleanup

        # Wrap OSError in InfraConnectionError per ONEX error handling guidelines
        # Deferred import to avoid circular dependency (utils -> errors -> utils)
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="write_atomic_bytes",
            target_name=str(path),
        )
        raise InfraConnectionError(
            f"Atomic write failed for '{path}'",
            context=context,
            error_type=type(e).__name__,
            errno=getattr(e, "errno", None),
        ) from e


async def write_atomic_bytes_async(
    path: Path,
    data: bytes,
    *,
    temp_prefix: str = "",
    temp_suffix: str = ".tmp",
    correlation_id: UUID | None = None,
) -> int:
    """Write bytes to a file atomically (async version).

    This is a thin async wrapper around :func:`write_atomic_bytes` that uses
    ``asyncio.to_thread()`` to run the synchronous implementation in a thread
    pool. This prevents blocking the event loop during file I/O.

    All logic is delegated to :func:`write_atomic_bytes` - this function exists
    only to provide an async interface.

    Args:
        path: Target file path. Parent directory must exist.
        data: Bytes to write to the file.
        temp_prefix: Optional prefix for the temporary file name.
        temp_suffix: Suffix for the temporary file name. Defaults to ".tmp".
        correlation_id: Optional correlation ID for ONEX logging.

    Returns:
        Number of bytes written to the file.

    Raises:
        InfraConnectionError: If the file cannot be written (permissions, disk full,
            etc.). The underlying OSError is chained via ``from e``.

    Example:
        >>> import asyncio
        >>> from pathlib import Path
        >>> from omnibase_infra.utils.util_atomic_file import write_atomic_bytes_async
        >>>
        >>> async def example():
        ...     path = Path("/tmp/test_async.txt")
        ...     return await write_atomic_bytes_async(path, b"async data")
        >>>
        >>> asyncio.run(example())
        10

    See Also:
        :func:`write_atomic_bytes`: The synchronous canonical implementation.
    """
    return await asyncio.to_thread(
        write_atomic_bytes,
        path,
        data,
        temp_prefix=temp_prefix,
        temp_suffix=temp_suffix,
        correlation_id=correlation_id,
    )


__all__: list[str] = [
    "write_atomic_bytes",
    "write_atomic_bytes_async",
]
