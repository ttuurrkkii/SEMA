"""
Memory Writer Node — LangGraph version

Records success/failure to STSC/DBRFPA after the pipeline completes.
Only stores logic failures (bad approach, wrong output, code errors).
Skips infrastructure errors (API 404/500, network timeouts, rate limits).
"""

import re
from state import OESAState
from nodes.searcher import stsc, dbrfpa
from token_tracker import tracker

# Patterns that indicate infrastructure errors — NOT logic failures
INFRA_ERROR_PATTERNS = [
    r"Error code: [45]\d\d",         # HTTP 4xx/5xx
    r"status.code.*[45]\d\d",
    r"Internal Server Error",
    r"Connection.*refused",
    r"Connection.*reset",
    r"Connection.*timed?\s*out",
    r"Timeout",
    r"ECONNREFUSED",
    r"ECONNRESET",
    r"ETIMEDOUT",
    r"rate.limit",
    r"Rate.Limit",
    r"Too Many Requests",
    r"quota.*exceeded",
    r"No allowed providers",
    r"API key",
    r"api_key",
    r"authentication",
    r"unauthorized",
    r"forbidden",
    r"DNS.*resolution",
    r"SSL.*error",
    r"certificate.*error",
]


def _is_infra_error(error: str) -> bool:
    """Return True if the error is infrastructure, not logic."""
    for pattern in INFRA_ERROR_PATTERNS:
        if re.search(pattern, error, re.IGNORECASE):
            return True
    return False


def memory_writer_node(state: OESAState) -> dict:
    """LangGraph node: write results to memory."""
    error = state.get("error")

    eval_passed = state.get("eval_passed", False)
    eval_score = state.get("eval_score", 0.0)
    eval_feedback = state.get("eval_feedback", "")

    if error:
        if _is_infra_error(error):
            print(f"\n  [MEMORY] Skipped — infrastructure error (not stored)")
            print(f"           {error[:100]}")
        else:
            agents = state.get("agents", [])
            agent_names = [a["name"] for a in agents] if agents else []
            approach = f"{state.get('orchestrator_logic', 'Sequential')} | agents: {agent_names}" if agent_names else "Pipeline failed before planning"

            dbrfpa.add(
                task=state["task"],
                domain=state.get("domain", "agent_builder"),
                failure_reason=error,
                approach_attempted=approach,
            )
            print(f"\n  [MEMORY] DBRFPA written — logic failure recorded")
            print(f"  DBRFPA entries: {len(dbrfpa)}")
    elif not eval_passed and eval_feedback:
        # Solver generated but failed evaluation
        agents = state.get("agents", [])
        agent_names = [a["name"] for a in agents] if agents else []
        approach = f"{state.get('orchestrator_logic', 'Sequential')} | agents: {agent_names}"

        dbrfpa.add(
            task=state["task"],
            domain=state.get("domain", "agent_builder"),
            failure_reason=f"Eval failed (score={eval_score}): {eval_feedback}",
            approach_attempted=approach,
        )
        print(f"\n  [MEMORY] DBRFPA written — eval failure recorded (score={eval_score})")
        print(f"  DBRFPA entries: {len(dbrfpa)}")
    else:
        agents = state.get("agents", [])
        agent_names = [a["name"] for a in agents]
        score_str = f" | eval_score={eval_score}" if eval_score > 0 else ""
        stsc.add(
            task=state["task"],
            domain="agent_builder",
            approach_used=f"{state.get('orchestrator_logic', 'Sequential')} | agents: {agent_names}{score_str}",
            result_summary=f"Generated {state.get('workflow_name', '?')} ({len(agents)} agents) → {state.get('output_file', '?')}",
        )
        print(f"\n  [MEMORY] STSC written — success recorded (eval_score={eval_score})")
        print(f"  STSC entries: {len(stsc)}")

    token_usage = tracker.totals()
    print(f"\n  [TOKENS] {tracker.summary_line()}")
    for node, u in token_usage["by_node"].items():
        print(f"    {node:14s}: {u['prompt_tokens']:>5} prompt + {u['completion_tokens']:>4} completion ({u['calls']} call{'s' if u['calls'] != 1 else ''})")

    return {"token_usage": token_usage}
