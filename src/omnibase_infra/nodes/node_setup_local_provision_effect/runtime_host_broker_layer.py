# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The one decision of whether a lane's compose file list gains the per-host broker layer (OMN-24509).

A lane that runs its broker per host lists its overlay plus the per-host broker layer file, so each
host's compose reaches its own broker listeners instead of another container's. The layer is listed
after the lane's overlay, and only for the overlay the contract names, when the host declares either
of the two advertised broker address variables the contract names. This module holds that decision
once: the shell's compose-files resolver and the deploy agent's executor both call ``layer_for``
here, so the two file lists cannot differ.

The overlay file name, the layer file name, the two variable names and the address pattern come only
from the contract's ``config.stability_host_broker_layer`` block, read by ``declaration()`` — no
value lives in this file. A declared variable that is empty, only whitespace, or not a
comma-separated list of listener://host:port entries is refused by name, never read as no layer. A
half declaration (exactly one of the two variables declared) still applies the layer, so compose
refuses the missing address by name.

The standalone deploy agent loads this file by path, so it imports the standard library and pyyaml
only.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

#: The declaration's own keys: the schema this module reads the contract block through.
_DECLARATION_KEYS: tuple[str, ...] = (
    "lane_overlay_prefix",
    "lane_overlay_suffix",
    "layer_file",
    "kafka_addr_env",
    "pandaproxy_addr_env",
    "address_pattern",
)


class StabilityHostBrokerLayerError(ValueError):
    """A declared address cannot be used, or the contract declaration cannot be read."""


@cache
def declaration(contract: Path) -> dict[str, str]:
    """The node contract's ``config.stability_host_broker_layer`` in the ``contract`` file, whole or not at all.

    Every consumer here takes its overlay file, layer file, variable names and address pattern from
    this block, so a missing or mistyped key refuses instead of falling back to a value in code.
    """
    try:
        document = yaml.safe_load(contract.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StabilityHostBrokerLayerError(
            f"{contract}: cannot be read: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise StabilityHostBrokerLayerError(f"{contract}: not YAML: {exc}") from exc
    config = document.get("config") if isinstance(document, dict) else None
    block = (
        config.get("stability_host_broker_layer") if isinstance(config, dict) else None
    )
    if not isinstance(block, dict):
        raise StabilityHostBrokerLayerError(
            f"{contract}: no config.stability_host_broker_layer block"
        )
    declared: dict[str, str] = {}
    for key in _DECLARATION_KEYS:
        value = block.get(key)
        if not isinstance(value, str) or not value.strip():
            raise StabilityHostBrokerLayerError(
                f"{contract}: config.stability_host_broker_layer.{key}: "
                "missing or not a non-empty string"
            )
        declared[key] = value
    return declared


def layer_for(
    overlay_filename: str,
    environ: Mapping[str, str],
    declared: Mapping[str, str],
) -> str | None:
    """The per-host broker layer file for ``overlay_filename``'s lane, or ``None`` when it gains none.

    ``declared`` is the ``declaration()`` mapping. A variable counts as declared when its name is a
    key of ``environ``, even at an empty value; an empty, whitespace-only or malformed value is
    refused by name and never read as no layer. The overlay is matched by equality, not by pattern.
    """
    if (
        overlay_filename
        != declared["lane_overlay_prefix"] + declared["lane_overlay_suffix"]
    ):
        return None
    address_envs = (declared["kafka_addr_env"], declared["pandaproxy_addr_env"])
    declared_envs = [name for name in address_envs if name in environ]
    if not declared_envs:
        return None
    for name in declared_envs:
        value = environ[name]
        if not value.strip():
            raise StabilityHostBrokerLayerError(f"{name} is declared but empty")
        if re.fullmatch(declared["address_pattern"], value) is None:
            raise StabilityHostBrokerLayerError(
                f"{name} is not a comma-separated list of listener://host:port entries"
            )
    return declared["layer_file"]


def layered[ConfigT: Any](execution: ModuleType, config: ConfigT) -> ConfigT:
    """``config`` with the layer listed after the lane overlay when the host declares it, else unchanged.

    ``execution`` is the deploy agent's executor module (the sibling modules' convention): it names
    the checkout the agent runs from, the clone the compose files live in and the operator store,
    and ``config`` is one lane's ``ModelLaneConfig``. The environment is the two sources the lane's
    compose commands read, the process environment and then the operator store. A refused
    declaration raises, so a deploy that cannot list the layer stops rather than recreating the
    broker without it.
    """
    checkout = Path(str(execution.__file__)).resolve().parents[3]
    # The node's package is the node's directory: this module's own dotted name locates its contract.
    declared = declaration(
        checkout.joinpath("src", *__name__.split(".")[:-1], "contract.yaml")
    )
    overlay = declared["lane_overlay_prefix"] + declared["lane_overlay_suffix"]
    files = config.compose_files
    if overlay not in {Path(path).name for path in files}:
        return config
    layer = layer_for(
        overlay, {**os.environ, **execution.operator_store_overlay()}, declared
    )
    if layer is None:
        return config
    return config.model_copy(
        update={
            "compose_files": (*files, f"{execution.deploy_repo_dir()}/docker/{layer}")
        }
    )


def main(argv: list[str], environ: Mapping[str, str]) -> int:
    """The shell's entry: ``layer`` prints the layer file name, ``describe`` the decision and its inputs."""
    if len(argv) != 3 or argv[0] not in ("layer", "describe"):
        sys.stderr.write(
            f"usage: {Path(__file__).name} layer|describe <contract_file> <overlay_filename>\n"
        )
        return 64
    command, contract_file, overlay_filename = argv
    try:
        declared = declaration(Path(contract_file))
        layer = layer_for(overlay_filename, environ, declared)
    except StabilityHostBrokerLayerError as exc:
        sys.stderr.write(f"compose-files: {exc}\n")
        return 2
    if command == "layer":
        sys.stdout.write(f"{layer}\n" if layer is not None else "")
        return 0
    sys.stdout.write(
        f"layer_file={layer or 'none'}\n"
        f"kafka_addr={environ.get(declared['kafka_addr_env'], 'unset')}\n"
        f"pandaproxy_addr={environ.get(declared['pandaproxy_addr_env'], 'unset')}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], os.environ))
