"""MCP server (stdio) so local agents (Claude Code, Codex, Hermes, ...) can call Nirnay offline.

  python -m nirnay.mcp_server

Tools:
  decide(state, questions, min_confidence?)  -> the same JSON as Nirnay.predict
  classify(text, labels, instructions?)      -> quick single choice question
  yes_no(text, question)                     -> P(yes)
"""
import json
from typing import Any, Dict, List, Optional

from .engine import load

_model = None


def model():
    global _model
    if _model is None:
        _model = load()
    return _model


def main():
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    mcp = Server("nirnay")

    @mcp.tool()
    def decide(state: Any, questions: Dict[str, Dict[str, Any]], min_confidence: Optional[float] = None) -> str:
        """Answer typed questions (choice / score / noul) about a state in one pass. Returns answers with
        calibrated probabilities; never generates text. Question spec: {"type": "choice"|"score"|"noul",
        "instructions": str, "criteria": {label: description} | [levels] | {"true": .., "false": ..}}."""
        return json.dumps(model().predict(state, questions, min_confidence=min_confidence), ensure_ascii=False)

    @mcp.tool()
    def classify(text: str, labels: List[str], instructions: str = "Which option fits best?") -> str:
        """Pick one label for a text. Returns {"choice", "probabilities", "answer_confidence"}."""
        r = model().predict(text, {"c": {"type": "choice", "instructions": instructions,
                                         "criteria": {l: None for l in labels}}})["answers"]["c"]
        return json.dumps({k: r[k] for k in ("choice", "probabilities", "answer_confidence")}, ensure_ascii=False)

    @mcp.tool()
    def yes_no(text: str, question: str) -> str:
        """Probability that the answer to a yes/no question about the text is yes."""
        r = model().predict(text, {"q": {"type": "noul", "instructions": question}})["answers"]["q"]
        return json.dumps({"p_yes": r["noul"], "answer": r["noul"] >= 0.5}, ensure_ascii=False)

    mcp.run()


if __name__ == "__main__":
    main()
