# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ``onex delegate`` flags that mark a run as not a user's run (OMN-23163).

``--traffic-class`` and ``--source-surface`` are stamped on the request's typed
provenance by :func:`~omnibase_infra.models.delegation.delegate_request_payload.build_delegate_provenance`.
``--probe-ladder`` (OMN-23164) names a committed ladder file and rides with
them, since only synthetic traffic may carry one. The command line module only
shrinks, so the option declarations live here and are applied as one decorator.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import click

from omnibase_core.enums.enum_delegation_traffic_class import (
    EnumDelegationTrafficClass,
)

__all__ = ["traffic_options"]


def traffic_options[F: Callable[..., object]](command: F) -> F:
    """Add ``--traffic-class`` and ``--source-surface`` to a click command."""
    surface = click.option(
        "--source-surface",
        "source_surface",
        type=str,
        default=None,
        help=(
            "Named producer surface stamped on the request provenance, such "
            "as the probe or experiment that issued the run. Required with "
            "--traffic-class synthetic."
        ),
    )
    traffic = click.option(
        "--traffic-class",
        "traffic_class",
        type=click.Choice([member.value for member in EnumDelegationTrafficClass]),
        default=None,
        help=(
            "Traffic class stamped on the request provenance: 'synthetic' "
            "marks a probe or experiment run as not a user's run and requires "
            "--source-surface; omit to leave the run unclassified."
        ),
    )
    ladder = click.option(
        "--probe-ladder",
        "probe_ladder",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help=(
            "Committed ladder file (YAML or JSON) sent with the request so a "
            "probe walks the rungs it names. Refused unless --traffic-class "
            "synthetic is given."
        ),
    )
    return traffic(surface(ladder(command)))
