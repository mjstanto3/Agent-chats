"""Tests for the Multi-Agent LLM Debate Framework (mock backend only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from multi_agent_debate.agents.agent_factory import create_agent, load_agents_from_config
from multi_agent_debate.agents.base_agent import Agent
from multi_agent_debate.blackboard.debate_state import DebateState
from multi_agent_debate.conversation.conversation_manager import ConversationManager
from multi_agent_debate.conversation.message_schema import Message
from multi_agent_debate.moderator.moderator_agent import ModeratorAgent
from multi_agent_debate.orchestrator.debate_orchestrator import DebateOrchestrator
from multi_agent_debate.synthesis.synthesizer import Synthesizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_topic() -> str:
    return "Is remote work more productive than office work?"


@pytest.fixture
def debate_state(sample_topic) -> DebateState:
    return DebateState(topic=sample_topic)


@pytest.fixture
def conversation(sample_topic) -> ConversationManager:
    return ConversationManager(topic=sample_topic)


@pytest.fixture
def mock_proponent() -> Agent:
    return create_agent("Alice", "proponent", backend="mock")


@pytest.fixture
def mock_critic() -> Agent:
    return create_agent("Bob", "critic", backend="mock")


@pytest.fixture
def mock_agents() -> list:
    return [
        create_agent("Alice", "proponent", backend="mock"),
        create_agent("Bob", "critic", backend="mock"),
        create_agent("Carol", "researcher", backend="mock"),
        create_agent("Dave", "analyst", backend="mock"),
        create_agent("Eve", "synthesizer", backend="mock"),
    ]


# ---------------------------------------------------------------------------
# Message schema
# ---------------------------------------------------------------------------

class TestMessage:
    def test_default_id_and_timestamp(self):
        msg = Message(content="hello", agent="TestAgent", round=1)
        assert msg.id.startswith("msg_")
        assert msg.timestamp  # non-empty
        assert msg.reply_to is None

    def test_to_dict_keys(self):
        msg = Message(content="test content", agent="Proponent", round=2)
        d = msg.to_dict()
        assert set(d.keys()) == {"id", "round", "agent", "reply_to", "content", "timestamp", "state_update"}

    def test_to_markdown_contains_content(self):
        msg = Message(content="Some argument here.", agent="Critic", round=3)
        md = msg.to_markdown()
        assert "Critic" in md
        assert "Some argument here." in md
        assert "Round 3" in md

    def test_reply_to_in_markdown(self):
        msg = Message(content="Reply.", agent="Analyst", round=2, reply_to="msg_abc123")
        md = msg.to_markdown()
        assert "msg_abc123" in md


# ---------------------------------------------------------------------------
# DebateState (Blackboard)
# ---------------------------------------------------------------------------

class TestDebateState:
    def test_initial_state(self, sample_topic, debate_state):
        assert debate_state.topic == sample_topic
        assert debate_state.claims == []
        assert debate_state.objections == []

    def test_apply_update_adds_claims(self, debate_state):
        debate_state.apply_update({"new_claims": ["Claim A", "Claim B"]})
        assert "Claim A" in debate_state.claims
        assert "Claim B" in debate_state.claims

    def test_apply_update_deduplicates(self, debate_state):
        debate_state.apply_update({"new_claims": ["Claim A"]})
        debate_state.apply_update({"new_claims": ["Claim A"]})
        assert debate_state.claims.count("Claim A") == 1

    def test_apply_update_objections(self, debate_state):
        debate_state.apply_update({"new_objections": ["Objection 1"]})
        assert "Objection 1" in debate_state.objections

    def test_apply_update_candidate_answer(self, debate_state):
        debate_state.apply_update({"candidate_answer": "Final answer here."})
        assert debate_state.candidate_answer == "Final answer here."

    def test_apply_update_consensus(self, debate_state):
        debate_state.apply_update({"new_consensus": ["Both sides agree on X"]})
        assert "Both sides agree on X" in debate_state.consensus_points

    def test_to_dict_structure(self, debate_state):
        d = debate_state.to_dict()
        assert set(d.keys()) == {
            "topic", "claims", "objections", "open_questions",
            "consensus_points", "candidate_answer",
        }

    def test_summary_is_string(self, debate_state):
        summary = debate_state.summary()
        assert isinstance(summary, str)
        assert debate_state.topic in summary


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------

class TestConversationManager:
    def test_initial_state(self, sample_topic, conversation):
        assert conversation.topic == sample_topic
        assert conversation.round == 0
        assert conversation.messages == []

    def test_next_round_increments(self, conversation):
        assert conversation.next_round() == 1
        assert conversation.next_round() == 2

    def test_add_message(self, conversation):
        conversation.next_round()
        msg = Message(content="Hello!", agent="Alice", round=1)
        conversation.add_message(msg)
        assert len(conversation.messages) == 1
        assert conversation.get_last_message() == msg

    def test_get_messages_by_agent(self, conversation):
        conversation.next_round()
        m1 = Message(content="A says hi", agent="Alice", round=1)
        m2 = Message(content="B says hi", agent="Bob", round=1)
        conversation.add_message(m1)
        conversation.add_message(m2)
        assert conversation.get_messages_by_agent("Alice") == [m1]
        assert conversation.get_messages_by_agent("Bob") == [m2]

    def test_export_json(self, tmp_path, conversation, sample_topic):
        conversation.next_round()
        conversation.add_message(Message(content="Test", agent="Alice", round=1))
        out = tmp_path / "transcript.json"
        conversation.export_json(out)
        data = json.loads(out.read_text())
        assert data["topic"] == sample_topic
        assert len(data["messages"]) == 1

    def test_export_markdown(self, tmp_path, conversation):
        conversation.next_round()
        conversation.add_message(Message(content="Markdown test", agent="Alice", round=1))
        out = tmp_path / "transcript.md"
        conversation.export_markdown(out)
        content = out.read_text()
        assert "# Debate Transcript" in content
        assert "Markdown test" in content


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgent:
    def test_repr(self, mock_proponent):
        r = repr(mock_proponent)
        assert "Alice" in r
        assert "proponent" in r

    def test_generate_message_returns_message(self, mock_proponent, conversation, debate_state):
        conversation.next_round()
        msg = mock_proponent.generate_message(conversation, debate_state)
        assert isinstance(msg, Message)
        assert msg.agent == "Alice"
        assert msg.content

    def test_propose_argument_returns_message(self, mock_proponent, sample_topic):
        msg = mock_proponent.propose_argument(sample_topic)
        assert isinstance(msg, Message)
        assert msg.content

    def test_critique_argument_returns_message(self, mock_critic):
        msg = mock_critic.critique_argument("AI is always beneficial.")
        assert isinstance(msg, Message)
        assert msg.content

    def test_respond_to_message(self, mock_critic, conversation, debate_state):
        conversation.next_round()
        target = Message(content="AI regulation is good.", agent="Alice", round=1)
        conversation.add_message(target)
        response = mock_critic.respond_to_message(target, conversation, debate_state)
        assert isinstance(response, Message)
        assert response.reply_to == target.id

    def test_state_update_in_message(self, mock_proponent, conversation, debate_state):
        conversation.next_round()
        msg = mock_proponent.generate_message(conversation, debate_state)
        # Mock proponent always includes state_update
        assert isinstance(msg.state_update, dict)

    def test_malformed_llm_response_handled(self, conversation, debate_state):
        """Agent should gracefully handle non-JSON LLM responses."""
        agent = create_agent("TestAgent", "proponent", backend="mock")
        # Monkey-patch _llm to return plain text
        agent._llm = lambda _: "This is plain text, not JSON."
        conversation.next_round()
        msg = agent.generate_message(conversation, debate_state)
        assert msg.content == "This is plain text, not JSON."
        assert msg.state_update == {}


# ---------------------------------------------------------------------------
# AgentFactory
# ---------------------------------------------------------------------------

class TestAgentFactory:
    def test_load_agents_from_config(self):
        config_path = ROOT / "multi_agent_debate" / "config" / "agents.yaml"
        if not config_path.exists():
            pytest.skip("agents.yaml not found")
        agents = load_agents_from_config(config_path, backend="mock")
        assert len(agents) >= 1
        for agent in agents:
            assert isinstance(agent, Agent)
            assert agent.name
            assert agent.role

    def test_load_agents_with_prompts(self):
        config_path = ROOT / "multi_agent_debate" / "config" / "agents.yaml"
        prompts_dir = ROOT / "multi_agent_debate" / "prompts" / "agent_prompts"
        if not config_path.exists():
            pytest.skip("agents.yaml not found")
        agents = load_agents_from_config(config_path, prompts_dir=prompts_dir, backend="mock")
        assert len(agents) >= 1
        # Prompts should contain the agent name (placeholder replaced)
        for agent in agents:
            assert agent.name in agent.system_prompt

    def test_create_agent_default_prompt(self):
        agent = create_agent("Tester", "critic", backend="mock")
        assert agent.name == "Tester"
        assert agent.role == "critic"
        assert "Tester" in agent.system_prompt


# ---------------------------------------------------------------------------
# ModeratorAgent
# ---------------------------------------------------------------------------

class TestModerator:
    def _make_moderator(self, agents: list) -> ModeratorAgent:
        return ModeratorAgent(
            agent_roles={a.name: a.role for a in agents},
            backend="mock",
            max_rounds=6,
        )

    def test_initial_decision_is_proponent(self, mock_agents, conversation, debate_state):
        mod = self._make_moderator(mock_agents)
        conversation.next_round()
        decision = mod.decide(conversation, debate_state)
        # No claims, no objections → should pick proponent
        assert decision.next_agent in [a.name for a in mock_agents if a.role == "proponent"]
        assert not decision.end_debate

    def test_routes_to_critic_on_objections(self, mock_agents, conversation, debate_state):
        debate_state.apply_update({"new_objections": ["Some objection"]})
        mod = self._make_moderator(mock_agents)
        conversation.next_round()
        decision = mod.decide(conversation, debate_state)
        assert decision.next_agent in [a.name for a in mock_agents if a.role == "critic"]

    def test_routes_to_researcher_on_open_questions(self, mock_agents, conversation, debate_state):
        debate_state.apply_update({"new_claims": ["A claim"]})
        debate_state.apply_update({"new_questions": ["What is the evidence?"]})
        mod = self._make_moderator(mock_agents)
        conversation.next_round()
        decision = mod.decide(conversation, debate_state)
        assert decision.next_agent in [a.name for a in mock_agents if a.role == "researcher"]

    def test_routes_to_analyst_on_multiple_claims(self, mock_agents, conversation, debate_state):
        debate_state.apply_update({"new_claims": ["Claim 1", "Claim 2"]})
        mod = self._make_moderator(mock_agents)
        conversation.next_round()
        decision = mod.decide(conversation, debate_state)
        assert decision.next_agent in [a.name for a in mock_agents if a.role == "analyst"]

    def test_ends_debate_at_max_rounds(self, mock_agents, conversation, debate_state):
        mod = ModeratorAgent(
            agent_roles={a.name: a.role for a in mock_agents},
            backend="mock",
            max_rounds=2,
        )
        for _ in range(3):
            conversation.next_round()
        decision = mod.decide(conversation, debate_state)
        assert decision.end_debate

    def test_ends_on_consensus(self, mock_agents, conversation, debate_state):
        debate_state.apply_update({"new_consensus": ["Agreement on X"]})
        # No open questions → convergence
        mod = self._make_moderator(mock_agents)
        conversation.next_round()
        decision = mod.decide(conversation, debate_state)
        assert decision.end_debate

    def test_decision_to_dict(self, mock_agents, conversation, debate_state):
        mod = self._make_moderator(mock_agents)
        conversation.next_round()
        d = mod.decide(conversation, debate_state).to_dict()
        assert "next_agent" in d
        assert "goal" in d
        assert "end_debate" in d


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class TestSynthesizer:
    def test_stub_synthesis(self, sample_topic, conversation, debate_state):
        synth = Synthesizer(backend="mock")
        debate_state.apply_update({"new_claims": ["Claim A"]})
        debate_state.apply_update({"candidate_answer": "The final answer."})
        result = synth.synthesize(conversation, debate_state)
        assert "Claim A" in result
        assert "The final answer." in result

    def test_stub_synthesis_no_data(self, sample_topic, conversation, debate_state):
        synth = Synthesizer(backend="mock")
        result = synth.synthesize(conversation, debate_state)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# DebateOrchestrator – end-to-end
# ---------------------------------------------------------------------------

class TestDebateOrchestrator:
    def test_full_debate_runs(self, tmp_path, sample_topic, mock_agents):
        orchestrator = DebateOrchestrator(
            topic=sample_topic,
            agents=mock_agents,
            max_rounds=4,
            output_dir=tmp_path,
            backend="mock",
        )
        final_answer = orchestrator.run()
        assert isinstance(final_answer, str)
        assert len(final_answer) > 0

    def test_transcripts_exported(self, tmp_path, sample_topic, mock_agents):
        orchestrator = DebateOrchestrator(
            topic=sample_topic,
            agents=mock_agents,
            max_rounds=3,
            output_dir=tmp_path,
            backend="mock",
        )
        orchestrator.run()
        output_files = list(tmp_path.iterdir())
        assert any(f.suffix == ".json" for f in output_files)
        assert any(f.suffix == ".md" for f in output_files)
        assert any(f.suffix == ".txt" for f in output_files)

    def test_json_transcript_valid(self, tmp_path, sample_topic, mock_agents):
        orchestrator = DebateOrchestrator(
            topic=sample_topic,
            agents=mock_agents,
            max_rounds=3,
            output_dir=tmp_path,
            backend="mock",
        )
        orchestrator.run()
        json_files = list(tmp_path.glob("*.json"))
        assert json_files
        data = json.loads(json_files[0].read_text())
        assert data["topic"] == sample_topic
        assert "messages" in data
        assert len(data["messages"]) >= 1

    def test_debate_respects_max_rounds(self, tmp_path, sample_topic, mock_agents):
        max_rounds = 3
        orchestrator = DebateOrchestrator(
            topic=sample_topic,
            agents=mock_agents,
            max_rounds=max_rounds,
            output_dir=tmp_path,
            backend="mock",
        )
        orchestrator.run()
        assert orchestrator.conversation.round <= max_rounds + 1  # +1 for synthesis round

    def test_blackboard_updated_during_debate(self, tmp_path, sample_topic, mock_agents):
        orchestrator = DebateOrchestrator(
            topic=sample_topic,
            agents=mock_agents,
            max_rounds=4,
            output_dir=tmp_path,
            backend="mock",
        )
        orchestrator.run()
        # After debate there should be at least some blackboard entries
        state = orchestrator.debate_state
        total_items = (
            len(state.claims) + len(state.objections) +
            len(state.open_questions) + len(state.consensus_points)
        )
        assert total_items > 0

    def test_unknown_agent_does_not_crash(self, tmp_path, sample_topic, monkeypatch):
        """Orchestrator should stop gracefully if moderator names an unknown agent."""
        from multi_agent_debate.moderator.moderator_agent import ModeratorDecision

        agents = [create_agent("Alice", "proponent", backend="mock")]
        orchestrator = DebateOrchestrator(
            topic=sample_topic,
            agents=agents,
            max_rounds=2,
            output_dir=tmp_path,
            backend="mock",
        )

        # Force the moderator to always return an unknown agent name
        monkeypatch.setattr(
            orchestrator.moderator,
            "decide",
            lambda conv, state: ModeratorDecision(
                next_agent="nonexistent_agent_xyz",
                goal="test goal",
                end_debate=False,
            ),
        )

        # Should exit gracefully (moderator picks unknown agent → loop breaks immediately)
        final_answer = orchestrator.run()
        assert isinstance(final_answer, str)
        # No agent ever produced a message – the loop broke before add_message was called
        assert len(orchestrator.conversation.messages) == 0
        # Blackboard should remain in its initial empty state
        assert orchestrator.debate_state.claims == []
        assert orchestrator.debate_state.objections == []
