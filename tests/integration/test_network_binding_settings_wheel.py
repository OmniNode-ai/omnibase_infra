# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise network binding consumers from the wheel shipped to the runtime."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_installed_wheel_loads_network_bindings_and_binds_loopback(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    wheels = tmp_path / "wheels"
    built = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(wheels)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next(wheels.glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)

    # -I excludes the source checkout and caller PYTHONPATH; import the extracted
    # wheel ahead of the venv's dependency packages and editable source finder.
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import socket
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from omnibase_infra.models.network_binding import model_network_binding_settings
from omnibase_infra.models.network_binding import load_network_binding_settings
from omnibase_infra.runtime.security_metadata_validator import SecurityMetadataValidator
from omnibase_infra.runtime.service_kernel import main
from omnibase_infra.services.post_merge.config import ConfigPostMergeConsumer

assert Path(model_network_binding_settings.__file__).is_relative_to(sys.argv[1])
assert callable(main)
settings = load_network_binding_settings()
config = ConfigPostMergeConsumer(kafka_bootstrap_servers="broker.test:9092", _env_file=None)
assert config.health_check_host == settings.loopback_address
with socket.socket() as server:
    server.bind((config.health_check_host, 0))
    assert server.getsockname()[0] == settings.loopback_address
validator = SecurityMetadataValidator()
assert validator.validate_domains([f"example.com:{settings.tcp_port_min}"]) == []
assert validator.validate_domains([f"example.com:{settings.tcp_port_max + 1}"])
print("wheel-network-binding-readback: settings loaded; kernel imported; loopback bound; port bounds enforced")
""",
            str(installed),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "wheel-network-binding-readback:" in probe.stdout
