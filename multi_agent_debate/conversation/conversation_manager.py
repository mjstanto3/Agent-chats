"""Conversation Manager – stores the full transcript and supports export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from multi_agent_debate.conversation.message_schema import Message

logger = logging.getLogger(__name__)


class ConversationManager:
    """Stores all debate messages and supports transcript export."""

    def __init__(self, topic: str) -> None:
        self.topic: str = topic
        self.round: int = 0
        self.messages: List[Message] = []

    def next_round(self) -> int:
        """Advance the round counter and return the new round number."""
        self.round += 1
        return self.round

    def add_message(self, message: Message) -> None:
        """Append a message to the transcript."""
        self.messages.append(message)
        logger.info("[Round %d] %s: %s", message.round, message.agent, message.content[:120])

    def get_last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None

    def get_messages_by_agent(self, agent_name: str) -> List[Message]:
        return [m for m in self.messages if m.agent == agent_name]

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "total_rounds": self.round,
            "total_messages": len(self.messages),
            "messages": [m.to_dict() for m in self.messages],
        }

    def export_json(self, path: Path) -> None:
        """Write full transcript as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Transcript exported to %s", path)

    def export_markdown(self, path: Path) -> None:
        """Write full transcript as Markdown."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Debate Transcript\n",
            f"**Topic:** {self.topic}\n",
            f"**Total rounds:** {self.round}  |  **Messages:** {len(self.messages)}\n",
            "---\n",
        ]
        for msg in self.messages:
            lines.append(msg.to_markdown())
            lines.append("\n---\n")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info("Transcript exported to %s", path)
