"""Synthesizer – generates the final refined answer from the debate state."""

from __future__ import annotations

import logging

from multi_agent_debate.blackboard.debate_state import DebateState
from multi_agent_debate.conversation.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

_SYNTHESIS_STUB = (
    "After careful deliberation the debate reached the following conclusion:\n\n"
    "**Strongest claims:** {claims}\n\n"
    "**Key objections addressed:** {objections}\n\n"
    "**Remaining open questions:** {questions}\n\n"
    "**Points of consensus:** {consensus}\n\n"
    "**Final recommendation:** {candidate}"
)


class Synthesizer:
    """
    Produces the final answer summarising:

    * strongest claims
    * key objections and how they were handled
    * resolved and remaining disagreements
    * remaining uncertainty
    * final recommendation
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.5,
        backend: str = "openai",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.backend = backend

    def synthesize(
        self,
        conversation: ConversationManager,
        debate_state: DebateState,
    ) -> str:
        """Return the final synthesis as a human-readable string."""
        if self.backend == "mock":
            return self._stub_synthesis(debate_state)
        return self._llm_synthesis(conversation, debate_state)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _llm_synthesis(
        self,
        conversation: ConversationManager,
        debate_state: DebateState,
    ) -> str:
        try:
            import openai  # type: ignore
        except ImportError:
            logger.warning("openai not installed; using stub synthesis.")
            return self._stub_synthesis(debate_state)

        full_transcript = "\n".join(
            f"[Round {m.round}] {m.agent}: {m.content}"
            for m in conversation.messages
        )
        system_prompt = (
            "You are the synthesis expert for a structured debate. "
            "Your job is to produce a concise, well-reasoned final answer "
            "that captures the outcome of the debate."
        )
        user_prompt = (
            f"Topic: {debate_state.topic}\n\n"
            f"Full transcript:\n{full_transcript}\n\n"
            f"Final blackboard state:\n{debate_state.summary()}\n\n"
            "Please write the final synthesis covering:\n"
            "1. The strongest claims made\n"
            "2. Key objections and how they were handled\n"
            "3. Points of consensus reached\n"
            "4. Remaining uncertainty\n"
            "5. Final recommendation\n"
        )

        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Synthesizer LLM call failed (%s); using stub.", exc)
            return self._stub_synthesis(debate_state)

    # ------------------------------------------------------------------
    # Stub / mock path
    # ------------------------------------------------------------------

    def _stub_synthesis(self, debate_state: DebateState) -> str:
        claims_str = "; ".join(debate_state.claims) or "none recorded"
        objections_str = "; ".join(debate_state.objections) or "none recorded"
        questions_str = "; ".join(debate_state.open_questions) or "none remaining"
        consensus_str = "; ".join(debate_state.consensus_points) or "none recorded"
        candidate = debate_state.candidate_answer or "No explicit candidate answer was produced."
        return _SYNTHESIS_STUB.format(
            claims=claims_str,
            objections=objections_str,
            questions=questions_str,
            consensus=consensus_str,
            candidate=candidate,
        )
