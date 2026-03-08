"""Debate Orchestrator – runs the full debate loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from multi_agent_debate.agents.base_agent import Agent
from multi_agent_debate.blackboard.debate_state import DebateState
from multi_agent_debate.conversation.conversation_manager import ConversationManager
from multi_agent_debate.moderator.moderator_agent import ModeratorAgent
from multi_agent_debate.synthesis.synthesizer import Synthesizer

logger = logging.getLogger(__name__)


class DebateOrchestrator:
    """
    Coordinates the full debate lifecycle:

    1. Initialise topic, agents, and state.
    2. Loop: moderator selects agent → agent generates message →
       append to transcript → update blackboard.
    3. On convergence / max-rounds: run Synthesizer.
    4. Export transcript artefacts.
    """

    def __init__(
        self,
        topic: str,
        agents: List[Agent],
        max_rounds: int = 10,
        output_dir: str | Path = "outputs",
        backend: str = "openai",
        moderator_model: str = "gpt-4o",
        synthesizer_model: str = "gpt-4o",
    ) -> None:
        self.topic = topic
        self.agents: Dict[str, Agent] = {a.name: a for a in agents}
        self.max_rounds = max_rounds
        self.output_dir = Path(output_dir)
        self.backend = backend

        self.conversation = ConversationManager(topic)
        self.debate_state = DebateState(topic=topic)

        agent_roles = {a.name: a.role for a in agents}
        self.moderator = ModeratorAgent(
            agent_roles=agent_roles,
            model=moderator_model,
            backend=backend,
            max_rounds=max_rounds,
        )
        self.synthesizer = Synthesizer(
            model=synthesizer_model,
            backend=backend,
        )

        self.final_answer: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Execute the debate and return the final synthesised answer."""
        logger.info("=== Debate started: %s ===", self.topic)
        logger.info("Agents: %s", list(self.agents.keys()))

        debate_active = True
        while debate_active:
            self.conversation.next_round()
            logger.info("--- Round %d ---", self.conversation.round)

            # Moderator decides
            decision = self.moderator.decide(self.conversation, self.debate_state)
            logger.info(
                "Moderator → %s | goal: %s | end: %s",
                decision.next_agent,
                decision.goal,
                decision.end_debate,
            )

            # Resolve agent
            agent = self._resolve_agent(decision.next_agent)
            if agent is None:
                logger.error("Moderator chose unknown agent '%s'; stopping.", decision.next_agent)
                break

            # Last message id for reply-to threading
            last_msg = self.conversation.get_last_message()
            reply_to = last_msg.id if last_msg else None

            # Agent generates message
            message = agent.generate_message(
                conversation=self.conversation,
                debate_state=self.debate_state,
                goal=decision.goal,
                reply_to=reply_to,
            )

            # Append to transcript
            self.conversation.add_message(message)

            # Update blackboard
            self.debate_state.apply_update(message.state_update)
            logger.debug("Blackboard:\n%s", self.debate_state.summary())

            # Check stopping conditions
            if decision.end_debate:
                logger.info("Moderator signalled end of debate.")
                debate_active = False
            elif self.conversation.round >= self.max_rounds:
                logger.info("Max rounds (%d) reached.", self.max_rounds)
                debate_active = False
            elif self._no_new_progress():
                logger.info("No new claims or objections in the last 2 rounds; ending debate.")
                debate_active = False

        # Synthesize final answer
        logger.info("=== Running synthesis ===")
        self.final_answer = self.synthesizer.synthesize(self.conversation, self.debate_state)
        logger.info("Synthesis complete.")

        # Export artefacts
        self._export()

        return self.final_answer

    # ------------------------------------------------------------------
    # Stopping conditions
    # ------------------------------------------------------------------

    def _no_new_progress(self) -> bool:
        """Return True if the last 2 rounds produced zero new blackboard items."""
        if len(self.conversation.messages) < 2:
            return False
        recent = self.conversation.messages[-2:]
        for msg in recent:
            su = msg.state_update
            if any([
                su.get("new_claims"),
                su.get("new_objections"),
                su.get("new_questions"),
                su.get("new_consensus"),
            ]):
                return False
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_agent(self, name: str) -> Optional[Agent]:
        """Look up agent by exact name, or by role as fallback."""
        if name in self.agents:
            return self.agents[name]
        # Fallback: try to match by role
        for agent in self.agents.values():
            if agent.role.lower() == name.lower():
                return agent
        return None

    def _export(self) -> None:
        """Save transcript as JSON and Markdown to the output directory."""
        slug = self.topic[:40].lower().replace(" ", "_").replace("/", "-")
        self.conversation.export_json(self.output_dir / f"{slug}_transcript.json")
        self.conversation.export_markdown(self.output_dir / f"{slug}_transcript.md")

        # Also write the final answer
        answer_path = self.output_dir / f"{slug}_final_answer.txt"
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        answer_path.write_text(self.final_answer or "", encoding="utf-8")
        logger.info("Final answer written to %s", answer_path)
