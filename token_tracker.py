"""
Token Tracker — records prompt/completion tokens per node across a pipeline run.

Supports both Anthropic (input_tokens/output_tokens) and OpenAI/OpenRouter
(prompt_tokens/completion_tokens) response metadata formats.

Usage:
    from token_tracker import tracker

    response = llm.invoke(messages)
    tracker.record("searcher", response)

    totals = tracker.totals()
    # {"prompt_tokens": 1200, "completion_tokens": 340, "total_tokens": 1540,
    #  "calls": 3, "by_node": {"searcher": {...}, "planner": {...}}}

    tracker.reset()  # call before each pipeline run
"""

import threading
from dataclasses import dataclass
from typing import List


@dataclass
class _NodeUsage:
    node: str
    prompt_tokens: int
    completion_tokens: int


def _extract_tokens(response) -> tuple:
    """Return (prompt_tokens, completion_tokens) from a LangChain AIMessage."""
    meta = getattr(response, "response_metadata", {}) or {}

    # Anthropic format: metadata["usage"]["input_tokens"] / ["output_tokens"]
    usage = meta.get("usage", {})
    if usage.get("input_tokens") is not None:
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    # OpenAI / OpenRouter format: metadata["token_usage"]["prompt_tokens"]
    tu = meta.get("token_usage", {})
    if tu.get("prompt_tokens") is not None:
        return tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0)

    return 0, 0


class TokenTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._records: List[_NodeUsage] = []

    def record(self, node: str, response) -> None:
        """Extract and store token usage from a LangChain LLM response."""
        prompt, completion = _extract_tokens(response)
        with self._lock:
            self._records.append(_NodeUsage(node, prompt, completion))

    def reset(self) -> None:
        """Clear all records — call at the start of each pipeline run."""
        with self._lock:
            self._records.clear()

    def totals(self) -> dict:
        """Return aggregated token counts, total and per-node."""
        with self._lock:
            records = list(self._records)

        prompt_total = sum(r.prompt_tokens for r in records)
        completion_total = sum(r.completion_tokens for r in records)

        by_node: dict = {}
        for r in records:
            entry = by_node.setdefault(
                r.node, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
            )
            entry["prompt_tokens"] += r.prompt_tokens
            entry["completion_tokens"] += r.completion_tokens
            entry["calls"] += 1

        return {
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "total_tokens": prompt_total + completion_total,
            "calls": len(records),
            "by_node": by_node,
        }

    def summary_line(self) -> str:
        """One-line summary for printing."""
        t = self.totals()
        return (
            f"tokens: {t['prompt_tokens']} prompt + "
            f"{t['completion_tokens']} completion = "
            f"{t['total_tokens']} total ({t['calls']} calls)"
        )


# Module-level singleton — imported and shared by all nodes
tracker = TokenTracker()
