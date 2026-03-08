"""Debate Blackboard – structured representation of the evolving reasoning state."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class DebateState:
    """Maintains the structured state of the debate across all turns."""

    topic: str
    claims: List[str] = field(default_factory=list)
    objections: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    consensus_points: List[str] = field(default_factory=list)
    candidate_answer: str = ""

    def apply_update(self, update: dict) -> None:
        """Merge a state_update dict produced by an agent into the blackboard."""
        new_claims = update.get("new_claims", [])
        new_objections = update.get("new_objections", [])
        new_questions = update.get("new_questions", [])
        new_consensus = update.get("new_consensus", [])
        candidate = update.get("candidate_answer")

        for claim in new_claims:
            if claim and claim not in self.claims:
                self.claims.append(claim)
                logger.debug("Blackboard: new claim → %s", claim)

        for obj in new_objections:
            if obj and obj not in self.objections:
                self.objections.append(obj)
                logger.debug("Blackboard: new objection → %s", obj)

        for q in new_questions:
            if q and q not in self.open_questions:
                self.open_questions.append(q)
                logger.debug("Blackboard: new question → %s", q)

        resolved_questions = update.get("resolved_questions", [])
        for q in resolved_questions:
            if q in self.open_questions:
                self.open_questions.remove(q)
                logger.debug("Blackboard: question resolved → %s", q)

        for cp in new_consensus:
            if cp and cp not in self.consensus_points:
                self.consensus_points.append(cp)
                logger.debug("Blackboard: new consensus point → %s", cp)

        if candidate:
            self.candidate_answer = candidate
            logger.debug("Blackboard: candidate_answer updated")

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "claims": copy.copy(self.claims),
            "objections": copy.copy(self.objections),
            "open_questions": copy.copy(self.open_questions),
            "consensus_points": copy.copy(self.consensus_points),
            "candidate_answer": self.candidate_answer,
        }

    def summary(self) -> str:
        """Return a compact human-readable summary of the current state."""
        lines = [
            f"Topic: {self.topic}",
            f"Claims ({len(self.claims)}): {', '.join(self.claims) or 'none'}",
            f"Objections ({len(self.objections)}): {', '.join(self.objections) or 'none'}",
            f"Open questions ({len(self.open_questions)}): {', '.join(self.open_questions) or 'none'}",
            f"Consensus ({len(self.consensus_points)}): {', '.join(self.consensus_points) or 'none'}",
            f"Candidate answer: {self.candidate_answer or '(none yet)'}",
        ]
        return "\n".join(lines)
