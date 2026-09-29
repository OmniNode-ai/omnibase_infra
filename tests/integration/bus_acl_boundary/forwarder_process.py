# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the gateway forwarder as its own process, as the deployed entrypoint does.

OMN-19928 (plan task S3). ``test_forwarder_refused_topic.py`` starts this file
with ``python`` in a separate process, so a refusal that kills the forwarder
shows up the way it does on a lab host: the process exits.

What is the same as ``python -m omnibase_infra.runtime.gateway_forwarder``:
the argument parser (``_build_parser``), the config loader
(``load_gateway_forwarder_runtime_config`` over the real node contract, the
broker-ref map and the lane-credential map), the secret resolver
(``AdapterEnvSecretStore``), the logging format, the SIGTERM/SIGINT shutdown
and ``run_gateway_forwarder`` itself with its ready file and health files.

What is different, and the only difference: the cloud leg's authentication.
The contract declares the cloud leg ``SASL_SSL`` with ``AWS_MSK_IAM``, and no
broker a CI job can start speaks MSK IAM. ``--cloud-leg-auth`` names a file
whose SCRAM settings replace the resolved cloud leg's authentication fields.
The replacement leg is validated on its own by ``ModelKafkaEventBusConfig``;
the one check it skips is the cross-leg rule that the resolved cloud mechanism
equals the declared one. RESIDUAL, the same one the OMN-18012 harness states:
this proves what the forwarder does when a broker refuses one of its topics,
not that it speaks IAM.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import yaml

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime import gateway_forwarder
from omnibase_infra.secret_stores.adapter_env_secret_store import (
    AdapterEnvSecretStore,
)

# The fields a cloud-leg auth override may set. Anything else in the file is
# refused, so the override cannot quietly change the broker address, the
# offset policy or anything else the real loader resolved.
_AUTH_FIELDS = frozenset(
    {
        "security_protocol",
        "sasl_mechanism",
        "sasl_plain_username",
        "sasl_plain_password",
        "msk_region",
    }
)


def _cloud_leg_with_auth(
    leg: ModelKafkaEventBusConfig, override_path: Path
) -> ModelKafkaEventBusConfig:
    raw: object = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--cloud-leg-auth must name a YAML mapping")
    override = {str(key): value for key, value in raw.items()}
    unexpected = sorted(set(override) - _AUTH_FIELDS)
    if unexpected:
        raise ValueError(f"--cloud-leg-auth may set only auth fields, got {unexpected}")
    fields = leg.model_dump(exclude=set(type(leg).model_computed_fields))
    return ModelKafkaEventBusConfig.model_validate({**fields, **override})


async def _run(argv: list[str]) -> None:
    parser = gateway_forwarder._build_parser()
    parser.add_argument("--cloud-leg-auth", type=Path, required=True)
    args = parser.parse_args(argv)

    config = gateway_forwarder.load_gateway_forwarder_runtime_config(
        args.config,
        broker_ref_map_path=args.broker_ref_map,
        lane_credential_map_path=args.lane_credential_map,
    )
    if config.cloud_bus is None:
        raise ValueError("the forwarder config resolved no cloud leg")
    config = config.model_copy(
        update={"cloud_bus": _cloud_leg_with_auth(config.cloud_bus, args.cloud_leg_auth)}
    )

    resolve_secret: Callable[[str], Awaitable[str | None]] = (
        AdapterEnvSecretStore().get_secret
    )
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)
    await gateway_forwarder.run_gateway_forwarder(
        config,
        shutdown_event=shutdown_event,
        resolve_secret=resolve_secret,
        ready_path=args.ready_file,
        egress_health_path=args.egress_health_file,
        lane_mirror_health_path=args.lane_mirror_health_file,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_run(sys.argv[1:]))


if __name__ == "__main__":
    main()
