"""Agent Factory – build Agent instances from YAML configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import yaml  # type: ignore

from multi_agent_debate.agents.base_agent import Agent

logger = logging.getLogger(__name__)

# Default prompt template used when no prompt file is found
_DEFAULT_SYSTEM_PROMPT = (
    "You are {name}, a {role} participating in a structured debate. "
    "Your goal is to {goal}. "
    "Always reason carefully and respond with well-structured arguments. "
    "When asked, return JSON with keys 'message' and 'state_update'."
)

_ROLE_GOALS: Dict[str, str] = {
    "proponent": "argue in favour of the strongest position, supporting it with evidence and logic",
    "critic": "identify weaknesses, challenge assumptions, and surface objections",
    "researcher": "provide factual grounding, cite evidence, and fill knowledge gaps",
    "analyst": "compare and reconcile conflicting claims, identify patterns and contradictions",
    "synthesizer": "summarise the debate, highlight agreements, and propose a final answer",
}


def load_agents_from_config(
    config_path: str | Path,
    prompts_dir: str | Path | None = None,
    backend: str = "openai",
) -> List[Agent]:
    """
    Load agent definitions from a YAML config file and return Agent instances.

    Parameters
    ----------
    config_path : path-like
        Path to ``agents.yaml``.
    prompts_dir : path-like, optional
        Directory containing ``<role>.txt`` prompt files.  When a file is
        found its contents replace the default system prompt.
    backend : str
        LLM backend to use for all agents ("openai" or "mock").
    """
    config_path = Path(config_path)
    prompts_dir = Path(prompts_dir) if prompts_dir else None

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    agents: List[Agent] = []
    for entry in config.get("agents", []):
        name: str = entry["name"]
        role: str = entry["role"]
        model: str = entry.get("model", "gpt-4o")
        temperature: float = float(entry.get("temperature", 0.7))

        system_prompt = _load_prompt(role, name, prompts_dir)

        agent = Agent(
            name=name,
            role=role,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            backend=backend,
        )
        agents.append(agent)
        logger.debug("Loaded agent: %s (%s)", name, role)

    logger.info("Loaded %d agents from %s", len(agents), config_path)
    return agents


def create_agent(
    name: str,
    role: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    system_prompt: str | None = None,
    backend: str = "openai",
) -> Agent:
    """
    Convenience factory to create a single Agent programmatically.

    If *system_prompt* is omitted a sensible default is generated from the role.
    """
    if system_prompt is None:
        goal = _ROLE_GOALS.get(role.lower(), f"contribute thoughtfully as a {role}")
        system_prompt = _DEFAULT_SYSTEM_PROMPT.format(name=name, role=role, goal=goal)

    return Agent(
        name=name,
        role=role,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_prompt(role: str, name: str, prompts_dir: Path | None) -> str:
    """Load a prompt from file if available, otherwise use the default."""
    if prompts_dir is not None:
        prompt_file = prompts_dir / f"{role.lower()}.txt"
        if prompt_file.exists():
            text = prompt_file.read_text(encoding="utf-8").strip()
            logger.debug("Loaded prompt for role '%s' from %s", role, prompt_file)
            return text.replace("{name}", name)

    goal = _ROLE_GOALS.get(role.lower(), f"contribute thoughtfully as a {role}")
    return _DEFAULT_SYSTEM_PROMPT.format(name=name, role=role, goal=goal)
