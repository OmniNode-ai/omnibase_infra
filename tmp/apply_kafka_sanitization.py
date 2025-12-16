#!/usr/bin/env python3
"""Apply ONEX compliance sanitization fixes to KafkaEventBus.

This script applies all necessary sanitization and error context fixes
to address PR #37 review feedback.
"""

import re
from pathlib import Path

def sanitize_kafka_event_bus():
    """Apply all sanitization fixes to kafka_event_bus.py."""
    file_path = Path(__file__).parent.parent / "src/omnibase_infra/event_bus/kafka_event_bus.py"

    with open(file_path, "r") as f:
        content = f.read()

    # Fix 1: Start method - TimeoutError (line ~450-460)
    content = re.sub(
        r'(\s+)context = ModelInfraErrorContext\(\s+transport_type=EnumInfraTransportType\.KAFKA,\s+operation="start",\s+target_name=f"kafka\.{self\._bootstrap_servers}",\s+correlation_id=uuid4\(\),\s+\)\s+logger\.warning\(\s+f"Timeout connecting to Kafka after {self\._timeout_seconds}s",\s+extra={"bootstrap_servers": self\._bootstrap_servers},\s+\)\s+raise InfraTimeoutError\(\s+f"Timeout connecting to Kafka after {self\._timeout_seconds}s",\s+context=context,\s+bootstrap_servers=self\._bootstrap_servers,',
        r'\1# Sanitize servers for safe logging (remove credentials)\n\1sanitized_servers = self._sanitize_bootstrap_servers(self._bootstrap_servers)\n\1context = ModelInfraErrorContext(\n\1    transport_type=EnumInfraTransportType.KAFKA,\n\1    operation="start",\n\1    target_name=f"kafka.{self._environment}",\n\1    correlation_id=uuid4(),\n\1)\n\1logger.warning(\n\1    f"Timeout connecting to Kafka after {self._timeout_seconds}s",\n\1    extra={"environment": self._environment},\n\1)\n\1raise InfraTimeoutError(\n\1    f"Timeout connecting to Kafka after {self._timeout_seconds}s",\n\1    context=context,\n\1    servers=sanitized_servers,',
        content,
        flags=re.DOTALL
    )

    # Fix 2: Start method - General Exception (line ~473-486)
    content = re.sub(
        r'(except Exception as e:\s+# Clean up producer on failure.*?\n.*?async with self\._producer_lock:.*?\n.*?self\._producer = None.*?\n.*?# Record failure.*?\n.*?self\._record_circuit_failure\(\).*?\n.*?)context = ModelInfraErrorContext\(\s+transport_type=EnumInfraTransportType\.KAFKA,\s+operation="start",\s+target_name=f"kafka\.{self\._bootstrap_servers}",\s+correlation_id=uuid4\(\),\s+\)\s+logger\.warning\(\s+f"Failed to connect to Kafka: {e}",\s+extra={\s+"bootstrap_servers": self\._bootstrap_servers,\s+"error": str\(e\),\s+},\s+\)\s+raise InfraConnectionError\(\s+f"Failed to connect to Kafka: {e}",\s+context=context,\s+bootstrap_servers=self\._bootstrap_servers,',
        r'\1# Sanitize servers for safe logging (remove credentials)\n                sanitized_servers = self._sanitize_bootstrap_servers(self._bootstrap_servers)\n                # Sanitize error message (remove potential sensitive details)\n                error_msg = str(e)\n                if "@" in error_msg or "password" in error_msg.lower():\n                    error_msg = "Connection failed (credentials sanitized)"\n                context = ModelInfraErrorContext(\n                    transport_type=EnumInfraTransportType.KAFKA,\n                    operation="start",\n                    target_name=f"kafka.{self._environment}",\n                    correlation_id=uuid4(),\n                )\n                logger.warning(\n                    f"Failed to connect to Kafka: {error_msg}",\n                    extra={\n                        "environment": self._environment,\n                        "error_type": type(e).__name__,\n                    },\n                )\n                raise InfraConnectionError(\n                    f"Failed to connect to Kafka: {error_msg}",\n                    context=context,\n                    servers=sanitized_servers,',
        content,
        flags=re.DOTALL
    )

    # Fix 3: _start_consumer_for_topic method (line ~925-932)
    content = re.sub(
        r'(except Exception as e:.*?context = ModelInfraErrorContext\(\s+transport_type=EnumInfraTransportType\.KAFKA,\s+operation="start_consumer",\s+target_name=f"kafka\.{topic}",\s+correlation_id=uuid4\(\),\s+\)\s+logger\.exception\(f"Failed to start consumer for topic {topic}"\)\s+raise InfraConnectionError\(\s+f"Failed to start consumer for topic {topic}",\s+context=context,\s+topic=topic,\s+)bootstrap_servers=self\._bootstrap_servers,',
        r'\1# Sanitized - topic is safe, no servers exposed',
        content,
        flags=re.DOTALL
    )

    # Fix 4: _check_circuit_breaker method (line ~1158)
    content = re.sub(
        r'target_name=f"kafka\.{self\._bootstrap_servers}",',
        r'target_name=f"kafka.{self._environment}",',
        content
    )

    # Fix 5: Logger extras with bootstrap_servers (multiple locations)
    # Success log (line ~437) - Keep this one as it's success, not error
    # But fix error logs
    content = re.sub(
        r'logger\.warning\(\s+f"Timeout connecting to Kafka after {self\._timeout_seconds}s",\s+extra={"bootstrap_servers": self\._bootstrap_servers},',
        r'logger.warning(\n                    f"Timeout connecting to Kafka after {self._timeout_seconds}s",\n                    extra={"environment": self._environment},',
        content
    )

    content = re.sub(
        r'logger\.warning\(\s+f"Circuit breaker opened after {self\._circuit_failure_count} failures",\s+extra={"bootstrap_servers": self\._bootstrap_servers},',
        r'logger.warning(\n                f"Circuit breaker opened after {self._circuit_failure_count} failures",\n                extra={"environment": self._environment},',
        content
    )

    # Fix 6: health_check method bootstrap_servers exposure (line ~1122) - This is OK for health_check
    # But let's sanitize it anyway for consistency
    content = re.sub(
        r'(return {\s+"healthy":.*?"group": self\._group,\s+)"bootstrap_servers": self\._bootstrap_servers,',
        r'\1"bootstrap_servers": self._sanitize_bootstrap_servers(self._bootstrap_servers),',
        content,
        flags=re.DOTALL
    )

    with open(file_path, "w") as f:
        f.write(content)

    print(f"✅ Applied sanitization fixes to {file_path}")
    print("Fixed:")
    print("  - Start method timeout error (sanitized servers, fixed target_name)")
    print("  - Start method connection error (sanitized servers and error messages)")
    print("  - Consumer start error (removed servers exposure)")
    print("  - Circuit breaker error (fixed target_name)")
    print("  - Logger warnings (sanitized server references)")
    print("  - Health check (sanitized servers)")

if __name__ == "__main__":
    sanitize_kafka_event_bus()
