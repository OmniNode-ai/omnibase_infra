#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# Wrapper: run topic-naming-lint against both contract YAML files and Python
# source files. (OMN-3259)
#
# Usage: invoked by pre-commit as a system-language hook.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINT="$SCRIPT_DIR/lint_topic_names.py"
RC=0

uv run --frozen python "$LINT" --scan-contracts src/omnibase_infra/nodes || RC=$?
uv run --frozen python "$LINT" --scan-python src/omnibase_infra || RC=$?

exit "$RC"
