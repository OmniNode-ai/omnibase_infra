# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A request to the listener holder (OMN-22379)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeListenerHolderRequest(BaseModel):
    """One holder operation.

    ``install_bind``: bind both listeners and put them in the systemd fd store.
    ``recover``: take the listeners back by name from LISTEN_FDS/LISTEN_FDNAMES.
    ``reboot_rebind``: after a host reboot, bind only when the install record
    reads ``completed`` and the ports are free.
    ``prepare_reexec``: dup2 the listeners onto descriptors 3 and 4 and set the
    LISTEN_* variables, immediately before ``os.execv``.
    ``reexec``: remap and exec transactionally; restore descriptors and LISTEN_*
    if exec fails, so the running agent can continue on its current image.
    ``drop``: install rollback; remove from the fd store and close.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    operation: Literal[
        "install_bind", "recover", "reboot_rebind", "prepare_reexec", "reexec", "drop"
    ] = Field(description="The holder operation.")
    executable: str = Field(
        default="", description="Executable for the transactional re-exec."
    )
    argv: tuple[str, ...] = Field(default=(), description="Arguments for the re-exec.")
    host: str = Field(default="::", description="Bind address (dual-stack).")
    main_port: int = Field(default=8085, ge=0, le=65535, description="runtime-main.")
    effects_port: int = Field(
        default=8086, ge=0, le=65535, description="runtime-effects."
    )
    backlog: int = Field(default=4096, ge=1, description="Listen backlog from policy.")
    install_record_path: str = Field(
        default="",
        description="JSON file whose 'state' reads 'completed' once an install is done.",
    )


__all__: list[str] = ["ModelRuntimeListenerHolderRequest"]
