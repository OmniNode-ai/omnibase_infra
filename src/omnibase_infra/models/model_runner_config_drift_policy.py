# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the image roll compares between a running runner and the synced Compose, and what it never compares."""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_runner_mount_type import EnumRunnerMountType
from omnibase_infra.models.model_runner_rendered_config import (
    ModelRunnerRenderedConfig,
)
from omnibase_infra.models.model_runner_running_config import (
    ModelRunnerRunningConfig,
)

# A volume the daemon made for an image ``VOLUME`` line is named by its 64 hex digit id and appears in no Compose file.
_ANONYMOUS_VOLUME = re.compile(r"[0-9a-f]{64}")


# ``kind, bind source, read-only``: the three facts of a mount that a Compose edit can change.
_MountFact = tuple[str, str, bool]


class ModelRunnerConfigDriftPolicy(BaseModel):
    """OMN-26105: a running runner differs from its synced Compose service when a mount or a declared env key does.

    ``env_keys`` is the whole declared set, sorted: adding a key is a code change that this model's tests see. It holds
    no secret and no value minted per render. The rendered ``RUNNER_TOKEN`` is a fresh registration token on every
    render, so a whole-environment comparison would recreate every online runner on every call (that killed a job on
    2026-09-28); the token, ``LOCAL_LLM_SHARED_SECRET`` and every undeclared key are never read off the host.

    Mounts compare in both directions by target, as kind, bind source and read-only flag. A named volume compares by
    target alone, because Compose prefixes its project to the name. Anonymous volumes belong to the image, not to
    Compose, and are ignored. Env compares only the declared keys that the rendered service sets.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    env_keys: tuple[str, ...] = (
        "ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE",
        "ACTIONS_RUNNER_HOOK_JOB_STARTED",
        "AGENT_TOOLSDIRECTORY",
        "GITHUB_ORG_URL",
        "KUBECONFIG",
        "ONEX_LAB_RUN_HOSTS",
        "ONEX_LANE_HOST",
        "RUNNER_GROUP",
        "RUNNER_LABELS",
        "RUNNER_NAME",
        "TMPDIR",
        "UV_CACHE_DIR",
        "UV_CONCURRENT_BUILDS",
        "UV_CONCURRENT_DOWNLOADS",
        "UV_CONCURRENT_INSTALLS",
        "UV_HTTP_TIMEOUT",
    )

    RENDERED_FILTER: ClassVar[str] = (
        "[.services[$n] | {volumes: (.volumes // []), tmpfs: (.tmpfs // []), environment: ((.environment // {})"
        " | with_entries(select(.key as $k | $keys | index($k))))}][0]"
    )
    RUNNING_FILTER: ClassVar[str] = (
        ".[0] | {mounts: (.Mounts // []), tmpfs: (.HostConfig.Tmpfs // {}), environment: ((.Config.Env // [])"
        ' | map(select(index("=") != null) | {key: .[:index("=")], value: .[index("=")+1:]}) | from_entries'
        " | with_entries(select(.key as $k | $keys | index($k))))}"
    )

    @staticmethod
    def _rendered_mounts(rendered: ModelRunnerRenderedConfig) -> dict[str, _MountFact]:
        mounts: dict[str, _MountFact] = {}
        for volume in rendered.volumes:
            if volume.type is EnumRunnerMountType.VOLUME and not volume.source:
                continue
            if volume.type is EnumRunnerMountType.TMPFS:
                mounts[volume.target] = ("tmpfs", "", False)
            else:
                source = (
                    volume.source if volume.type is EnumRunnerMountType.BIND else ""
                )
                mounts[volume.target] = (volume.type, source, volume.read_only)
        for entry in rendered.tmpfs:
            mounts[entry.split(":", 1)[0]] = ("tmpfs", "", False)
        return mounts

    @staticmethod
    def _running_mounts(running: ModelRunnerRunningConfig) -> dict[str, _MountFact]:
        mounts: dict[str, _MountFact] = {}
        for mount in running.mounts:
            if mount.kind == "volume" and _ANONYMOUS_VOLUME.fullmatch(mount.name):
                continue
            if mount.kind == "tmpfs":
                mounts[mount.target] = ("tmpfs", "", False)
            else:
                source = mount.source if mount.kind == "bind" else ""
                mounts[mount.target] = (mount.kind, source, not mount.writable)
        for target in running.tmpfs:
            mounts[target] = ("tmpfs", "", False)
        return mounts

    @classmethod
    def drift(cls, rendered_json: str, running_json: str) -> tuple[str, ...]:
        """Return ``mount:<target>`` and ``env:<KEY>`` labels for each difference; ``()`` means no drift.

        Raises ``ValueError`` when either document does not parse: an unreadable runner is never reported as in sync.
        """
        rendered = ModelRunnerRenderedConfig.model_validate_json(rendered_json)
        running = ModelRunnerRunningConfig.model_validate_json(running_json)
        want, have = cls._rendered_mounts(rendered), cls._running_mounts(running)
        labels = [
            f"mount:{target}"
            for target in sorted(set(want) | set(have))
            if want.get(target) != have.get(target)
        ]
        declared = cls().env_keys
        labels += [
            f"env:{key}"
            for key in declared
            if rendered.environment.get(key) is not None
            and rendered.environment[key] != running.environment.get(key)
        ]
        return tuple(labels)
