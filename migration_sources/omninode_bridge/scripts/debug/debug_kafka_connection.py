#!/usr/bin/env python3
"""Debug script to isolate aiokafka connection issues."""

import asyncio
import logging
import socket

from aiokafka import AIOKafkaProducer

# Reduce logging noise
logging.basicConfig(level=logging.WARNING)


async def test_connection():
    """Test different aiokafka connection approaches."""
    print("Testing aiokafka connection approaches...")

    # Test if we can resolve the hostname
    print("\n=== DNS Resolution Test ===")
    try:
        ip = socket.gethostbyname("omninode-bridge-redpanda")
        print(f"✅ omninode-bridge-redpanda resolves to: {ip}")
    except socket.gaierror as e:
        print(f"❌ omninode-bridge-redpanda DNS resolution failed: {e}")

    # Test raw TCP connection to both addresses
    print("\n=== TCP Connection Tests ===")
    for host, port in [("localhost", 29092), ("127.0.0.1", 29092)]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f"✅ TCP connection to {host}:{port} succeeded")
            else:
                print(f"❌ TCP connection to {host}:{port} failed with code {result}")
        except Exception as e:
            print(f"❌ TCP connection to {host}:{port} failed: {e}")

    # Test different bootstrap server formats
    test_configs = [
        ("localhost:29092", "Standard localhost"),
        ("127.0.0.1:29092", "IP address"),
        (["localhost:29092"], "List format"),
        (["127.0.0.1:29092"], "List with IP"),
    ]

    for bootstrap_servers, description in test_configs:
        print(f"\n=== {description}: {bootstrap_servers} ===")
        try:
            producer = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                request_timeout_ms=3000,
                api_version="auto",
                enable_idempotence=False,
                acks=1,
            )
            await producer.start()
            print(f"✅ {description} producer connected!")
            await producer.stop()
            break  # Stop on first success
        except Exception as e:
            print(f"❌ {description} producer failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(test_connection())
