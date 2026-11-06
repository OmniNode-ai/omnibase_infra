# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The client half of the work-ledger read node's wire contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class ModelWorkLedgerQueryClientConfig(BaseModel):
    """A dev-only caller, with its topics and exact granted reply-group prefix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane: Literal["dev"]
    command_topic: str = Field(min_length=1)
    completed_topic: str = Field(min_length=1)
    consumer_group_prefix: str = Field(min_length=16, pattern=r"\.$")
    queries: tuple[str, ...] = Field(min_length=1)
    #: Queries the node answers from work_ledger_state (omnimarket OMN-21948):
    #: a status, a page size and a watermark, and no freshness or parity.
    point_queries: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _point_queries_are_queries(self) -> "ModelWorkLedgerQueryClientConfig":
        unknown = set(self.point_queries) - set(self.queries)
        if unknown:
            raise ValueError(
                f"point_queries {sorted(unknown)} are not in the client's queries"
            )
        return self

    def answer_problem(
        self, kind: JsonValue, answer: dict[str, JsonValue]
    ) -> str | None:
        """Why the node's answer cannot be returned, or None when it can.

        A node refusal (status ``refused``) is the node's own sentence naming its
        bound and the bounded form of the request; it holds no database detail.
        A point answer must carry its own facts (status ok, watermark, page size);
        every other answer must carry freshness and parity.
        """
        if answer.get("status") == "refused":
            refusal = answer.get("refusal")
            if isinstance(refusal, dict) and isinstance(refusal.get("suggestion"), str):
                return f"Ledger node refused the request: {refusal['suggestion']}"
            return "Ledger node refused the request and named no bound"
        if kind in self.point_queries:
            if (
                answer.get("status") == "ok"
                and isinstance(answer.get("watermark"), str)
                and isinstance(answer.get("page_size"), int)
            ):
                return None
            return "Ledger point reply lacks status ok, watermark or page size"
        if isinstance(answer.get("freshness"), dict) and isinstance(
            answer.get("parity"), dict
        ):
            return None
        return "Ledger reply lacks freshness or parity"
