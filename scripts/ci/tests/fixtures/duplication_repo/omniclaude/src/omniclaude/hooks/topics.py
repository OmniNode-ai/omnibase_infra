from enum import StrEnum

class TopicBase(StrEnum):
    SESSION_STARTED = "onex.evt.omniclaude.session-started.v1"
    PROMPT_SUBMITTED = "onex.evt.omniclaude.prompt-submitted.v1"
