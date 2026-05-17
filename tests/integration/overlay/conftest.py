# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def overlay_dir(tmp_path: Path) -> Path:
    """Temp directory for overlay YAML files."""
    return tmp_path / "overlays"


@pytest.fixture
def sample_overlay_yaml(overlay_dir: Path) -> Path:
    """Write a valid overlay YAML and return its path."""
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / "overlay.yaml"
    overlay_data = {
        "overlay_version": "1.0.0",
        "environment": "test",
        "scope": "env",
        "transports": {
            "kafka": {
                "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
            "database": {
                "POSTGRES_HOST": "localhost",
                "POSTGRES_PORT": "5436",
                "POSTGRES_PASSWORD": "test_password",
            },
        },
    }
    overlay_path.write_text(yaml.safe_dump(overlay_data, sort_keys=True))
    overlay_path.chmod(0o600)
    return overlay_path


@pytest.fixture
def contracts_dir(tmp_path: Path) -> Path:
    """Temp directory with a minimal contract YAML requiring kafka + database."""
    cdir = tmp_path / "contracts"
    cdir.mkdir()
    contract = {
        "name": "test-node",
        "node_type": "EFFECT_GENERIC",
        "metadata": {"transport_type": "kafka"},
        "dependencies": [
            {"type": "environment", "key": "POSTGRES_HOST"},
            {"type": "environment", "key": "KAFKA_BOOTSTRAP_SERVERS"},
        ],
    }
    (cdir / "contract.yaml").write_text(yaml.safe_dump(contract))
    return cdir
