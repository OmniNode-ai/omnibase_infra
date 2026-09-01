#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Compatibility entry point; implementation is the durable Python runner.
set -euo pipefail
exec python3 /home/jonah/push-lanes/queue-runner.py "$@"
