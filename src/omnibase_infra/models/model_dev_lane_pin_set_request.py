# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to set one dev lane pin in the operator env store (OMN-18475)."""

from pydantic import BaseModel, ConfigDict


class ModelDevLanePinSetRequest(BaseModel):
    """Everything a set needs, all of it named by the caller.

    ``key`` and ``lane`` are plain strings on purpose: a prod key or a lane other
    than dev is a named refusal in the result, not a validation error that a
    caller could read as a malformed request. ``ledger_command`` is the argv
    prefix of the ledger writer; the ledger path and ``--append <row>`` are added
    by the handler. ``execute`` false validates and writes nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    env_file: str
    key: str
    value: str
    lane: str
    ledger_path: str
    ledger_command: tuple[str, ...]
    ledger_lane: str
    ticket: str
    execute: bool = False
    docker_cmd: str = "docker"
