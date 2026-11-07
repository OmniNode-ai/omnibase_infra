# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A minimal runtime process for the OMN-21229 SIGTERM case.

Started by ``test_subscription_lifecycle_real_broker_omn21229.py`` as a child
process. It does what the runtime kernel does on termination: a SIGTERM
handler closes the bus, which pauses every subscription, lets the in-flight
callback finish and stops. It prints one line per event so the parent can
sequence the signal and read the outcome:

``group=<consumer group>``  after the subscription is live
``started=<offset>``        when a callback begins
``completed=<offset>``      when a callback returns
"""

from __future__ import annotations

import asyncio
import signal
import sys

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


async def main(bootstrap: str, topic: str, group_id: str, seconds: float) -> None:
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=bootstrap,
            environment="local",
            timeout_seconds=30,
            auto_offset_reset="earliest",
            consumer_shutdown_drain_seconds=20.0,
        )
    )
    await bus.start()
    terminated = asyncio.Event()

    async def _callback(message: ModelEventMessage) -> None:
        print(f"started={message.offset}", flush=True)
        await asyncio.sleep(seconds)
        print(f"completed={message.offset}", flush=True)

    await bus.subscribe(topic, group_id=group_id, on_message=_callback)
    consumer = bus._group_consumers[(topic, group_id)]
    print(f"group={consumer._group_id}", flush=True)

    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, terminated.set)
    await terminated.wait()
    await bus.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])))
