# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where the runner installer reads the lab host table from and where the containers read it."""

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerLabHostTablePolicy(BaseModel):
    """OMN-26098: the declared source and the installer-owned copy of the lab host table.

    The declared source is the overlay projection on the runner host (owner-only, mode 0600, owned by the deploy user).
    A runner container cannot read that file, so each deploy copies it to ``copy_home_relative`` and writes the
    copy's absolute path to ``env_key`` in the host's ``docker/.env``; every runner compose mounts that path read-only
    at ``/etc/onex/lab_run_hosts.yaml``.

    Both paths are relative to the home directory of the SSH user on the runner host, resolved on that host. The request's
    ``lab_host_table_projection`` (``RUNNER_LAB_HOST_TABLE_PROJECTION`` in the deploying environment) replaces the
    projection path.

    ``reader_uid`` is the uid the Runner.Worker runs as: ``docker/runners/Dockerfile`` creates ``runner`` as uid 1001,
    the entrypoint starts as root and drops to it with gosu. The copy is owned by that uid and by the deploy user's
    group, mode 0440: the runner and the deploy user read it, no other host user does, and nobody writes it but root.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    env_key: str = "RUNNER_LAB_HOST_TABLE"
    projection_home_relative: str = (
        ".local/state/onex/overlay-store/lab_hosts/overlay.yaml"
    )
    copy_home_relative: str = ".omnibase/runners/lab-host-table/lab_run_hosts.yaml"
    reader_uid: int = Field(default=1001, ge=0)
    copy_mode: str = "440"
    copy_dir_mode: str = "750"
