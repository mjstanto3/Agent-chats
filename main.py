"""
Multi-Agent LLM Debate Framework – entry point.

Usage
-----
Run with the default mock backend (no API key required):

    python main.py

Run with OpenAI (requires OPENAI_API_KEY environment variable):

    python main.py --backend openai --topic "Is remote work better than office work?"

Options
-------
  --topic TEXT       The debate topic (default: hard-coded example)
  --backend TEXT     "mock" (offline) or "openai" (live LLM calls) [default: mock]
  --rounds INT       Maximum number of debate rounds [default: 6]
  --output-dir PATH  Directory for transcript artefacts [default: outputs]
  --config PATH      Path to agents.yaml [default: multi_agent_debate/config/agents.yaml]
  --prompts-dir PATH Path to agent prompt templates directory
  --log-level TEXT   Logging level (DEBUG/INFO/WARNING) [default: INFO]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml  # type: ignore


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _load_settings(settings_path: Path) -> dict:
    if settings_path.exists():
        with open(settings_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-agent LLM debate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topic",
        default="Should artificial intelligence be regulated by governments?",
        help="The debate topic",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["mock", "openai"],
        help="LLM backend: 'mock' for offline testing, 'openai' for live calls (default: from settings.yaml or 'mock')",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Maximum number of debate rounds (default: from settings.yaml or 6)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for transcript artefacts (default: from settings.yaml or 'outputs')",
    )
    parser.add_argument(
        "--config",
        default="multi_agent_debate/config/agents.yaml",
        help="Path to agents.yaml",
    )
    parser.add_argument(
        "--prompts-dir",
        default="multi_agent_debate/prompts/agent_prompts",
        help="Directory containing <role>.txt prompt files",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity",
    )

    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    logger = logging.getLogger("main")

    # Load settings (override defaults with settings.yaml values)
    settings_path = Path("multi_agent_debate/config/settings.yaml")
    settings = _load_settings(settings_path)

    backend = args.backend or settings.get("debate", {}).get("backend", "mock")
    max_rounds = args.rounds or settings.get("debate", {}).get("max_rounds", 6)
    output_dir = Path(args.output_dir or settings.get("debate", {}).get("output_dir", "outputs"))
    moderator_model = settings.get("llm", {}).get("moderator_model", "gpt-4o")
    synthesizer_model = settings.get("llm", {}).get("synthesizer_model", "gpt-4o")

    # Lazy imports (after sys.path is set up properly when run as a module)
    from multi_agent_debate.agents.agent_factory import load_agents_from_config
    from multi_agent_debate.orchestrator.debate_orchestrator import DebateOrchestrator

    config_path = Path(args.config)
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else None

    logger.info("Loading agents from %s", config_path)
    agents = load_agents_from_config(
        config_path=config_path,
        prompts_dir=prompts_dir,
        backend=backend,
    )

    orchestrator = DebateOrchestrator(
        topic=args.topic,
        agents=agents,
        max_rounds=max_rounds,
        output_dir=output_dir,
        backend=backend,
        moderator_model=moderator_model,
        synthesizer_model=synthesizer_model,
    )

    final_answer = orchestrator.run()

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_answer)
    print("=" * 70)
    print(f"\nTranscripts saved in: {output_dir}/")


if __name__ == "__main__":
    main()
