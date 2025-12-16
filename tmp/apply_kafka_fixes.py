#!/usr/bin/env python3
"""Apply ONEX compliance sanitization fixes to KafkaEventBus."""

from pathlib import Path

def apply_fixes():
    """Apply all sanitization fixes."""
    file_path = Path(__file__).parent.parent / "src/omnibase_infra/event_bus/kafka_event_bus.py"

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Track changes
    changes = []

    # Fix 1: Line ~450 - target_name with bootstrap_servers → environment
    for i, line in enumerate(lines):
        if 'target_name=f"kafka.{self._bootstrap_servers}"' in line and i < 500:
            lines[i] = line.replace(
                'target_name=f"kafka.{self._bootstrap_servers}"',
                'target_name=f"kafka.{self._environment}"'
            )
            changes.append(f"Line {i+1}: Fixed target_name in start() timeout error")

    # Fix 2: Line ~455 - logger extra with bootstrap_servers (in timeout handler)
    for i, line in enumerate(lines):
        if '"bootstrap_servers": self._bootstrap_servers' in line and 'extra=' in line and i < 500:
            # Check if this is the timeout error handler (near line 455)
            if i > 440 and i < 465:
                lines[i] = line.replace(
                    '"bootstrap_servers": self._bootstrap_servers',
                    '"environment": self._environment'
                )
                changes.append(f"Line {i+1}: Sanitized logger extra in timeout handler")

    # Fix 3: Line ~460 - bootstrap_servers kwarg in InfraTimeoutError
    for i, line in enumerate(lines):
        if 'bootstrap_servers=self._bootstrap_servers' in line and i > 455 and i < 465:
            # Add sanitization before this error
            indent = len(line) - len(line.lstrip())
            sanitization_line = " " * indent + "# Sanitize servers for safe logging (remove credentials)\n"
            san_var_line = " " * indent + "sanitized_servers = self._sanitize_bootstrap_servers(self._bootstrap_servers)\n"
            lines.insert(i, sanitization_line)
            lines.insert(i + 1, san_var_line)
            lines[i + 2] = lines[i + 2].replace(
                'bootstrap_servers=self._bootstrap_servers',
                'servers=sanitized_servers'
            )
            changes.append(f"Line {i+1}: Sanitized bootstrap_servers in InfraTimeoutError")
            break

    # Reload lines after insertion
    with open(file_path, "w") as f:
        f.writelines(lines)

    # Re-read for next fixes
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Fix 4: Line ~473 - target_name in connection error
    for i, line in enumerate(lines):
        if 'target_name=f"kafka.{self._bootstrap_servers}"' in line and i > 465 and i < 500:
            lines[i] = line.replace(
                'target_name=f"kafka.{self._bootstrap_servers}"',
                'target_name=f"kafka.{self._environment}"'
            )
            changes.append(f"Line {i+1}: Fixed target_name in start() connection error")

    # Fix 5: Line ~479 - logger extra in connection error
    for i, line in enumerate(lines):
        if '"bootstrap_servers": self._bootstrap_servers' in line and i > 470 and i < 490:
            lines[i] = line.replace(
                '"bootstrap_servers": self._bootstrap_servers',
                '"environment": self._environment'
            )
            changes.append(f"Line {i+1}: Sanitized logger extra in connection error")

    # Fix 6: Line ~486 - bootstrap_servers kwarg in InfraConnectionError (start method)
    for i, line in enumerate(lines):
        if 'bootstrap_servers=self._bootstrap_servers' in line and i > 480 and i < 495:
            # Add sanitization before this error
            indent = len(line) - len(line.lstrip())
            sanitization_line = " " * indent + "# Sanitize servers for safe logging (remove credentials)\n"
            san_var_line = " " * indent + "sanitized_servers = self._sanitize_bootstrap_servers(self._bootstrap_servers)\n"
            lines.insert(i, sanitization_line)
            lines.insert(i + 1, san_var_line)
            lines[i + 2] = lines[i + 2].replace(
                'bootstrap_servers=self._bootstrap_servers',
                'servers=sanitized_servers'
            )
            changes.append(f"Line {i+1}: Sanitized bootstrap_servers in InfraConnectionError (start)")
            break

    # Write intermediate
    with open(file_path, "w") as f:
        f.writelines(lines)

    # Re-read for consumer fixes
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Fix 7: Line ~932 - Remove bootstrap_servers from _start_consumer_for_topic error
    for i, line in enumerate(lines):
        if 'bootstrap_servers=self._bootstrap_servers' in line and i > 900 and i < 950:
            # Just remove this line by commenting it out or removing the parameter
            if ',' in lines[i-1] and 'topic=topic' in lines[i-1]:
                # Remove the trailing comma from previous line and remove this line
                lines[i-1] = lines[i-1].rstrip().rstrip(',') + '\n'
                lines[i] = lines[i].replace(
                    'bootstrap_servers=self._bootstrap_servers,',
                    '# bootstrap_servers removed for security (sanitization)'
                )
            changes.append(f"Line {i+1}: Removed bootstrap_servers from consumer error")

    # Fix 8: Line ~1158 - target_name in circuit breaker
    for i, line in enumerate(lines):
        if 'target_name=f"kafka.{self._bootstrap_servers}"' in line and i > 1140:
            lines[i] = line.replace(
                'target_name=f"kafka.{self._bootstrap_servers}"',
                'target_name=f"kafka.{self._environment}"'
            )
            changes.append(f"Line {i+1}: Fixed target_name in circuit breaker error")

    # Fix 9: Line ~1180 - Circuit breaker logger warning
    for i, line in enumerate(lines):
        if 'Circuit breaker opened after' in line and i > 1170:
            # Find the next line with bootstrap_servers in extra
            for j in range(i, min(i+3, len(lines))):
                if '"bootstrap_servers": self._bootstrap_servers' in lines[j]:
                    lines[j] = lines[j].replace(
                        '"bootstrap_servers": self._bootstrap_servers',
                        '"environment": self._environment'
                    )
                    changes.append(f"Line {j+1}: Sanitized circuit breaker logger")
                    break

    # Fix 10: Line ~1122 - health_check bootstrap_servers
    for i, line in enumerate(lines):
        if 'health_check' in ''.join(lines[max(0,i-10):i]) and '"bootstrap_servers": self._bootstrap_servers' in line and i > 1100 and i < 1130:
            lines[i] = line.replace(
                '"bootstrap_servers": self._bootstrap_servers',
                '"bootstrap_servers": self._sanitize_bootstrap_servers(self._bootstrap_servers)'
            )
            changes.append(f"Line {i+1}: Sanitized bootstrap_servers in health_check")

    # Write final version
    with open(file_path, "w") as f:
        f.writelines(lines)

    print("✅ Applied ONEX compliance fixes to KafkaEventBus")
    print(f"\n📝 Changes applied ({len(changes)}):")
    for change in changes:
        print(f"  - {change}")
    print("\n🔒 Security improvements:")
    print("  - Sanitized bootstrap_servers to remove credentials")
    print("  - Fixed target_name to use environment instead of servers")
    print("  - Removed server exposure from logger warnings")
    print("  - Sanitized health_check server output")

if __name__ == "__main__":
    apply_fixes()
