"""Moderator Agent – blackboard-driven routing of debate turns."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from multi_agent_debate.blackboard.debate_state import DebateState
from multi_agent_debate.conversation.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)


class ModeratorDecision:
    """The output of a single moderation step."""

    def __init__(self, next_agent: str, goal: str, end_debate: bool = False) -> None:
        self.next_agent = next_agent
        self.goal = goal
        self.end_debate = end_debate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_agent": self.next_agent,
            "goal": self.goal,
            "end_debate": self.end_debate,
        }

    def __repr__(self) -> str:
        return (
            f"ModeratorDecision(next_agent={self.next_agent!r}, "
            f"goal={self.goal!r}, end_debate={self.end_debate})"
        )


class ModeratorAgent:
    """
    The Moderator does **not** debate the topic.

    It inspects the transcript and blackboard after every turn and decides:

    * Which agent should speak next (based on gaps in the blackboard)
    * What that agent's goal for the turn should be
    * Whether the debate has converged and should end

    Routing heuristics
    ------------------
    * unresolved objections → Critic (or role matching "critic")
    * no claims yet, or fewer claims than objections → Proponent
    * open questions with no evidence → Researcher
    * conflicting claims → Analyst
    * debate converging (consensus_points present + few open_questions) → Synthesizer
    """

    def __init__(
        self,
        agent_roles: Dict[str, str],
        model: str = "gpt-4o",
        backend: str = "openai",
        max_rounds: int = 10,
        max_tokens: int = 256,
    ) -> None:
        """
        Parameters
        ----------
        agent_roles : dict
            Mapping of ``{agent_name: role}``.
        model : str
            LLM model to use when backend='openai'.
        backend : str
            "openai" or "mock".
        max_rounds : int
            Hard stop after this many rounds regardless of convergence.
        max_tokens : int
            Maximum tokens for each moderation decision response.
        """
        self.agent_roles = agent_roles  # {name: role}
        self.model = model
        self.backend = backend
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens

        # Build reverse mapping: role → first agent with that role
        self._role_to_agent: Dict[str, str] = {}
        for name, role in agent_roles.items():
            self._role_to_agent.setdefault(role.lower(), name)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def decide(
        self,
        conversation: ConversationManager,
        debate_state: DebateState,
    ) -> ModeratorDecision:
        """Inspect state and return the next moderation decision."""
        if self.backend == "openai":
            return self._decide_llm(conversation, debate_state)
        return self._decide_heuristic(conversation, debate_state)

    # ------------------------------------------------------------------
    # Heuristic (offline / mock) decision
    # ------------------------------------------------------------------

    def _decide_heuristic(
        self,
        conversation: ConversationManager,
        debate_state: DebateState,
    ) -> ModeratorDecision:
        """Rule-based routing used in mock/offline mode."""
        s = debate_state

        # Hard stop
        if conversation.round >= self.max_rounds:
            synthesizer = self._role_to_agent.get("synthesizer", _first_key(self.agent_roles))
            return ModeratorDecision(
                next_agent=synthesizer,
                goal="produce the final synthesis",
                end_debate=True,
            )

        # Convergence: consensus points exist and few open questions
        if s.consensus_points and len(s.open_questions) <= 1:
            synthesizer = self._role_to_agent.get("synthesizer", _first_key(self.agent_roles))
            return ModeratorDecision(
                next_agent=synthesizer,
                goal="summarise consensus and produce the final answer",
                end_debate=True,
            )

        # Unresolved objections and a critic is available
        if s.objections and self._role_to_agent.get("critic"):
            return ModeratorDecision(
                next_agent=self._role_to_agent["critic"],
                goal=f"challenge the weakest assumption given objection: {s.objections[-1]}",
            )

        # Open questions with no researcher yet
        if s.open_questions and self._role_to_agent.get("researcher"):
            return ModeratorDecision(
                next_agent=self._role_to_agent["researcher"],
                goal=f"provide evidence to answer: {s.open_questions[-1]}",
            )

        # Multiple claims suggest analysis needed
        if len(s.claims) >= 2 and self._role_to_agent.get("analyst"):
            return ModeratorDecision(
                next_agent=self._role_to_agent["analyst"],
                goal="reconcile conflicting claims and identify consensus",
            )

        # Default: start with the proponent (or the first agent)
        proponent = self._role_to_agent.get("proponent", _first_key(self.agent_roles))
        return ModeratorDecision(
            next_agent=proponent,
            goal="propose the strongest opening argument on the topic",
        )

    # ------------------------------------------------------------------
    # LLM-based decision
    # ------------------------------------------------------------------

    def _decide_llm(
        self,
        conversation: ConversationManager,
        debate_state: DebateState,
    ) -> ModeratorDecision:
        """Use the LLM to make the moderation decision."""
        try:
            import openai  # type: ignore
        except ImportError:
            logger.warning("openai not installed; falling back to heuristic moderator.")
            return self._decide_heuristic(conversation, debate_state)

        recent_msgs = conversation.messages[-5:]
        transcript_snippet = "\n".join(
            f"[{m.agent}]: {m.content[:200]}" for m in recent_msgs
        )
        available_agents = json.dumps(
            [{"name": k, "role": v} for k, v in self.agent_roles.items()]
        )

        system_prompt = (
            "You are the moderator of a structured LLM debate. "
            "Your role is to inspect the current debate state and decide "
            "which agent should speak next, what their goal should be, "
            "and whether the debate has converged.\n\n"
            "Respond ONLY with a JSON object matching this schema:\n"
            '{"next_agent": "<name>", "goal": "<one-sentence goal>", "end_debate": false}'
        )
        user_prompt = (
            f"Available agents:\n{available_agents}\n\n"
            f"Current blackboard:\n{debate_state.summary()}\n\n"
            f"Recent transcript:\n{transcript_snippet}\n\n"
            f"Current round: {conversation.round} / max {self.max_rounds}\n\n"
            "What is the next moderation decision?"
        )

        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            return ModeratorDecision(
                next_agent=data["next_agent"],
                goal=data.get("goal", ""),
                end_debate=bool(data.get("end_debate", False)),
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Moderator LLM call failed (%s); using heuristic.", exc)
            return self._decide_heuristic(conversation, debate_state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_key(d: dict) -> str:
    return next(iter(d))
