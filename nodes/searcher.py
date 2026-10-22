"""
Searcher Node — LangGraph version

Detects domain, queries STSC/DBRFPA memory,
and synthesizes approach hints for the Planner.
No internet access — relies only on memory and LLM knowledge.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from config import create_llm
from state import OESAState
from memory.stsc import STSC
from memory.dbrfpa import DBRFPA
from token_tracker import tracker


llm = create_llm()

# Shared memory instances
stsc = STSC()
dbrfpa = DBRFPA()


SYNTHESIS_PROMPT = """You are the Searcher Agent in the OESA Framework.
Analyze the incoming task and produce structured guidance for the Planner Agent.

Given:
- The raw task
- Past successes from memory (may be empty)
- Past failures from memory (may be empty)

Return a JSON object with exactly these fields:
{
  "domain": "<detected domain, e.g. math, reasoning, code, science, general>",
  "suggested_approach_hints": "<3-5 sentences advising the Planner on strategies>"
}

Be specific and actionable. Return ONLY the JSON object."""


def searcher_node(state: OESAState) -> dict:
    """LangGraph node: search memory, detect domain, produce hints."""
    task = state["task"]
    print("\n  [SEARCHER] Finding relevant patterns...")

    # 1. Query memory
    raw_successes = stsc.search(task, top_k=3)
    raw_failures = dbrfpa.search(task, top_k=3)

    if raw_successes:
        print(f"  [SEARCHER] Found {len(raw_successes)} relevant success(es) in memory")
    if raw_failures:
        print(f"  [SEARCHER] Found {len(raw_failures)} relevant failure(s) in memory")

    # 2. Build context
    parts = []
    if raw_successes:
        parts.append("Past Successes:")
        for s in raw_successes:
            parts.append(f"  - Task: {s['task']}, Approach: {s['approach_used']} (relevance: {s.get('relevance_score', '?')})")
    else:
        parts.append("Past Successes: none")

    if raw_failures:
        parts.append("Past Failures:")
        for f in raw_failures:
            parts.append(f"  - Task: {f['task']}, Failure: {f['failure_reason']} (relevance: {f.get('relevance_score', '?')})")
    else:
        parts.append("Past Failures: none")

    context_text = "\n".join(parts)

    # 3. LLM synthesizes domain + hints
    response = llm.invoke([
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=f"Task: {task}\n\n{context_text}\n\nAnalyze and return JSON."),
    ])
    tracker.record("searcher", response)

    # Parse response
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        parsed = {"domain": "math", "suggested_approach_hints": text}

    domain = parsed.get("domain", "math")
    hints = parsed.get("suggested_approach_hints", "")

    # Re-query memory with detected domain
    domain_successes = stsc.search(task, domain=domain, top_k=3)
    domain_failures = dbrfpa.search(task, domain=domain, top_k=3)

    print(f"  [SEARCHER] Domain: {domain}")
    print(f"  [SEARCHER] Hints: {hints[:200]}...")

    return {
        "domain": domain,
        "relevant_successes": domain_successes or raw_successes,
        "known_failures": domain_failures or raw_failures,
        "approach_hints": hints,
    }
