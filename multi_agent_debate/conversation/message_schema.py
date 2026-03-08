"""Message schema for the debate conversation transcript."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "msg") -> str:
    short = uuid.uuid4().hex[:6]
    return f"{prefix}_{short}"


@dataclass
class Message:
    """A single message in the debate transcript."""

    content: str
    agent: str
    round: int
    id: str = field(default_factory=lambda: _new_id("msg"))
    reply_to: Optional[str] = None
    timestamp: str = field(default_factory=_utc_now)

    # Optional structured update produced alongside the message
    state_update: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "round": self.round,
            "agent": self.agent,
            "reply_to": self.reply_to,
            "content": self.content,
            "timestamp": self.timestamp,
            "state_update": self.state_update,
        }

    def to_markdown(self) -> str:
        reply = f" *(replying to {self.reply_to})*" if self.reply_to else ""
        return (
            f"### [{self.id}] Round {self.round} – **{self.agent}**{reply}\n"
            f"*{self.timestamp}*\n\n"
            f"{self.content}\n"
        )
