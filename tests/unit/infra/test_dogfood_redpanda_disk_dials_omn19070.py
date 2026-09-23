# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19070: the dogfood broker bounds what an idle partition costs on disk.

Why this exists. On .105 the dogfood Redpanda broker and the arm64 CI runner
share one Docker Desktop virtual-machine disk (126 GB). The runner refuses
every job when that disk is under 5 GB free, and it was under that floor on
2026-09-21. The broker was the largest consumer: 1673 partition directories,
most of which carry almost no data. Each one still held a full 32 MiB, because
Redpanda preallocates the active segment in ``segment_fallocation_step``
chunks (33554432 by default), and a segment is only rolled after
``log_segment_ms`` (14 days by default) or ``log_segment_size`` (128 MiB).
Retention cannot delete an active segment, so an idle partition keeps its
32 MiB for as long as it exists.

Measured on a scratch single-node v24.2.7 broker on 2026-09-23. Ten
one-message topics held 32780 KiB each. With ``segment_fallocation_step`` set to
1048576, a new topic allocated 1024 KiB. Once the old segments passed the
lowered segment age, each rolled, the closed segment was truncated to its
real size, and the directory fell to 20 KiB. No broker restart was needed;
``rpk cluster config status`` read ``NEEDS-RESTART false``.

The dials, and the basis for each value:

* ``segment_fallocation_step`` = 1 MiB. With about 1700 partitions that bounds
  preallocation near 1.7 GiB, where 32 MiB bound it near 53 GiB.
* ``log_segment_ms`` = 1 day. Idle partitions roll within a day, which releases
  their preallocation and lets the 7-day retention act on them. Seven closed
  segments per active partition is a small file count.
* ``retention_bytes`` = 1 GiB per partition. The largest partitions on the
  .105 broker held 1.2 to 3.7 GiB on 2026-09-21, which is real data. A 1 GiB
  cap bounds that tail on a disk that must also keep the runner above 5 GB
  and a 12 GB lease floor for proof builds.

They are set by the ``redpanda-partition-cap`` one-shot, which runs on every
``up`` including against a warm volume. That is the case on .105, where the
volume predates the change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_COMPOSE = Path(__file__).resolve().parents[3] / "docker" / "docker-compose.dogfood.yml"

_EXPECTED = {
    "segment_fallocation_step": 1048576,
    "log_segment_ms": 86400000,
    "retention_bytes": 1073741824,
}


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Passthrough constructor for Docker Compose `!override` / `!reset` tags."""
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that unwraps compose override tags."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)
_ComposeLoader.add_constructor("!reset", _construct_compose_value)


def _init_settings() -> dict[str, str]:
    # _ComposeLoader extends SafeLoader; the extra constructors only unwrap
    # compose override tags.
    compose: dict[str, Any] = yaml.load(
        _COMPOSE.read_text(encoding="utf-8"),
        Loader=_ComposeLoader,  # noqa: S506
    )
    script = "\n".join(
        str(c) for c in compose["services"]["redpanda-partition-cap"]["command"]
    )
    settings: dict[str, str] = {}
    for line in script.splitlines():
        parts = line.split()
        if "cluster" in parts and "config" in parts and "set" in parts:
            key, value = parts[parts.index("set") + 1 : parts.index("set") + 3]
            settings[key] = value
    return settings


@pytest.mark.unit
def test_the_init_service_is_parsed() -> None:
    """Rule 16 positive control: the pre-existing partition cap is read back."""
    assert _init_settings()["topic_partitions_per_shard"] == "7000"


@pytest.mark.unit
@pytest.mark.parametrize(("key", "value"), sorted(_EXPECTED.items()))
def test_the_dogfood_broker_sets_each_disk_dial(key: str, value: int) -> None:
    assert _init_settings().get(key) == str(value)


@pytest.mark.unit
def test_preallocation_is_bounded_well_under_the_runner_floor() -> None:
    """1700 partitions at the configured step must stay under 2 GiB."""
    step = int(_init_settings()["segment_fallocation_step"])
    assert step % 4096 == 0
    assert 1700 * step < 2 * 1024**3
