# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from enum import StrEnum


class TopicBase(StrEnum):
    SESSION_STARTED = "onex.evt.omniclaude.session-started.v1"
    PROMPT_SUBMITTED = "onex.evt.omniclaude.prompt-submitted.v1"
