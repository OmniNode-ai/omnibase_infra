# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed media capability a lane overlay backend can serve (OMN-22756).

A node that needs speech, transcription, embeddings, a typed decision or video
asks the lane overlay for a capability and never for a host or port. Each
capability has exactly one endpoint shape, so a binding that declares a
capability and a URL ending in another capability's path is a declaration
error, not a runtime surprise.
"""

from __future__ import annotations

from enum import Enum, unique

__all__ = ["EnumBifrostLaneMediaCapability"]


@unique
class EnumBifrostLaneMediaCapability(str, Enum):
    """A media capability and the one endpoint path that serves it.

    Values:
        TTS: text to speech, served at ``/v1/audio/speech``.
        STT: speech to text, served at ``/v1/audio/transcriptions``.
        EMBED: embeddings, served at ``/v1/embeddings``.
        DECIDE: a typed decision over chat completions, served at
            ``/v1/chat/completions``.
        VIDEO: video generation, served at ``/v1/video/generations``.
    """

    TTS = "tts"
    STT = "stt"
    EMBED = "embed"
    DECIDE = "decide"
    VIDEO = "video"

    @property
    def endpoint_path(self) -> str:
        """The one URL path a backend serving this capability must end in."""
        return _ENDPOINT_PATHS[self]

    def __str__(self) -> str:
        """Return the plain value so messages read ``tts``, not the repr."""
        return self.value


_ENDPOINT_PATHS: dict[EnumBifrostLaneMediaCapability, str] = {
    EnumBifrostLaneMediaCapability.TTS: "/v1/audio/speech",
    EnumBifrostLaneMediaCapability.STT: "/v1/audio/transcriptions",
    EnumBifrostLaneMediaCapability.EMBED: "/v1/embeddings",
    EnumBifrostLaneMediaCapability.DECIDE: "/v1/chat/completions",
    EnumBifrostLaneMediaCapability.VIDEO: "/v1/video/generations",
}
