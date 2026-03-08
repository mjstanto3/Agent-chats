"""Base Agent – LLM participant with a specific reasoning role."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from multi_agent_debate.blackboard.debate_state import DebateState
from multi_agent_debate.conversation.conversation_manager import ConversationManager
from multi_agent_debate.conversation.message_schema import Message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM backend helpers
# ---------------------------------------------------------------------------

def _call_openai(
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call OpenAI chat completion and return the text content."""
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package is required when backend='openai'. "
            "Install it with: pip install openai"
        ) from exc

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        logger.warning(
            "OpenAI response message.content was None; returning empty string instead."
        )
        return ""
    return content


def _call_mock(system_prompt: str, user_prompt: str, role: str) -> str:
    """Deterministic stub used when backend='mock' (testing / offline mode)."""
    stub_map = {
        "proponent": (
            '{"message": "I propose that this approach is valid because it is well-supported '
            'by evidence and logic.", "state_update": {"new_claims": ["The approach is evidence-based"], '
            '"new_objections": [], "new_questions": ["What counter-evidence exists?"]}}'
        ),
        "critic": (
            '{"message": "I challenge the previous claim because it lacks empirical grounding.", '
            '"state_update": {"new_claims": [], "new_objections": ["Claim lacks empirical grounding"], '
            '"new_questions": ["Where is the empirical data?"]}}'
        ),
        "researcher": (
            '{"message": "Research suggests there are multiple perspectives on this topic.", '
            '"state_update": {"new_claims": ["Multiple perspectives exist per research"], '
            '"new_objections": [], "new_questions": []}}'
        ),
        "analyst": (
            '{"message": "Analyzing the claims, there appears to be a partial convergence.", '
            '"state_update": {"new_claims": [], "new_objections": [], '
            '"new_questions": [], "new_consensus": ["Partial convergence identified"]}}'
        ),
        "synthesizer": (
            '{"message": "Based on the debate, the strongest supported position is as follows.", '
            '"state_update": {"new_claims": [], "new_objections": [], "new_questions": [], '
            '"candidate_answer": "The debate converged on a well-supported, nuanced conclusion."}}'
        ),
    }
    role_key = role.lower()
    return stub_map.get(role_key, (
        '{"message": "I acknowledge the previous arguments and add my perspective.", '
        '"state_update": {"new_claims": ["Additional perspective noted"], '
        '"new_objections": [], "new_questions": []}}'
    ))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    An LLM participant in the debate with a specific reasoning role.

    Parameters
    ----------
    name : str
        Display name for the agent (e.g. "Alice").
    role : str
        Functional role label used for prompt routing and moderator logic
        (e.g. "proponent", "critic", "researcher", "analyst", "synthesizer").
    system_prompt : str
        The static system / persona prompt sent to the LLM.
    model : str
        LLM model identifier (e.g. "gpt-4o").
    temperature : float
        Sampling temperature for generation.
    backend : str
        "openai" (default) or "mock" (offline stub for testing).
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        backend: str = "openai",
    ) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.backend = backend

    # ------------------------------------------------------------------
    # Internal LLM call
    # ------------------------------------------------------------------

    def _llm(self, user_prompt: str) -> str:
        """Dispatch to the configured backend and return raw text."""
        if self.backend == "mock":
            return _call_mock(self.system_prompt, user_prompt, self.role)
        return _call_openai(self.model, self.temperature, self.system_prompt, user_prompt)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Parse the JSON envelope that agents are expected to return.

        Falls back gracefully when the LLM returns plain text instead of JSON.
        """
        raw = raw.strip()
        # Try to extract a JSON object from the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        # Graceful fallback
        logger.warning("%s returned non-JSON response; wrapping as plain message.", self.name)
        return {"message": raw, "state_update": {}}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_message(
        self,
        conversation: ConversationManager,
        debate_state: DebateState,
        goal: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> Message:
        """
        Generate the next contribution for the ongoing debate.

        The agent receives the full conversation transcript and the current
        blackboard state as context, then produces a message together with
        structured blackboard updates.
        """
        context = self._build_context(conversation, debate_state, goal)
        raw = self._llm(context)
        parsed = self._parse_response(raw)

        message_text = parsed.get("message", "")
        state_update = parsed.get("state_update", {})

        msg = Message(
            content=message_text,
            agent=self.name,
            round=conversation.round,
            reply_to=reply_to,
            state_update=state_update,
        )
        return msg

    def propose_argument(self, topic: str) -> Message:
        """
        Open the debate with an initial argument on the topic.

        Returns a standalone Message (not yet linked to a ConversationManager).
        """
        prompt = (
            f"The debate topic is: {topic}\n\n"
            "Please open the debate with your initial argument.\n"
            "Respond in JSON with the schema:\n"
            '{"message": "...", "state_update": {"new_claims": [], '
            '"new_objections": [], "new_questions": []}}'
        )
        raw = self._llm(prompt)
        parsed = self._parse_response(raw)
        return Message(
            content=parsed.get("message", ""),
            agent=self.name,
            round=1,
            state_update=parsed.get("state_update", {}),
        )

    def critique_argument(self, claim: str) -> Message:
        """
        Produce a critique of a specific claim.

        Returns a standalone Message.
        """
        prompt = (
            f"Please critique the following claim:\n\n{claim}\n\n"
            "Respond in JSON with the schema:\n"
            '{"message": "...", "state_update": {"new_claims": [], '
            '"new_objections": [], "new_questions": []}}'
        )
        raw = self._llm(prompt)
        parsed = self._parse_response(raw)
        return Message(
            content=parsed.get("message", ""),
            agent=self.name,
            round=0,
            state_update=parsed.get("state_update", {}),
        )

    def respond_to_message(
        self,
        target_message: Message,
        conversation: ConversationManager,
        debate_state: DebateState,
    ) -> Message:
        """
        Generate a direct response to a specific message in the transcript.
        """
        prompt = (
            f"You are responding directly to the following message "
            f"(id={target_message.id}) from {target_message.agent}:\n\n"
            f"{target_message.content}\n\n"
            f"Current blackboard state:\n{debate_state.summary()}\n\n"
            "Respond in JSON with the schema:\n"
            '{"message": "...", "state_update": {"new_claims": [], '
            '"new_objections": [], "new_questions": []}}'
        )
        raw = self._llm(prompt)
        parsed = self._parse_response(raw)
        return Message(
            content=parsed.get("message", ""),
            agent=self.name,
            round=conversation.round,
            reply_to=target_message.id,
            state_update=parsed.get("state_update", {}),
        )

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(
        self,
        conversation: ConversationManager,
        debate_state: DebateState,
        goal: Optional[str],
    ) -> str:
        """Build the full user prompt including transcript and blackboard."""
        recent = conversation.messages[-10:]  # last 10 messages for context
        transcript_lines = []
        for m in recent:
            transcript_lines.append(f"[{m.id}] {m.agent}: {m.content}")
        transcript_str = "\n".join(transcript_lines) if transcript_lines else "(no messages yet)"

        goal_str = f"\nYour goal for this turn: {goal}" if goal else ""

        return (
            f"Debate topic: {debate_state.topic}\n\n"
            f"Current blackboard state:\n{debate_state.summary()}\n\n"
            f"Recent conversation:\n{transcript_str}\n"
            f"{goal_str}\n\n"
            "Please contribute your next message to the debate.\n"
            "Respond in JSON with the schema:\n"
            '{"message": "...", "state_update": {"new_claims": [], '
            '"new_objections": [], "new_questions": [], '
            '"new_consensus": [], "candidate_answer": ""}}'
        )

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r}, model={self.model!r})"
