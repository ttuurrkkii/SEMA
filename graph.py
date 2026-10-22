"""
OESA Agent Builder — LangGraph Version

A StateGraph pipeline with evaluation feedback loop:

  Searcher → Planner → Executor → Evaluator ──→ MemoryWriter → END
                          ↑           │
                          └── retry ──┘

Usage:
    python graph.py
    > Enter math domain: algebra word problems
"""

import sys
import io
import os

try:
    sys.stdout = io.TextIOWrapper(open(1, 'wb', closefd=False), encoding='utf-8', errors='replace')
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

from langgraph.graph import StateGraph, END

from state import OESAState
from nodes.searcher import searcher_node
from nodes.planner import planner_node
from nodes.executor import executor_node
from nodes.evaluator import evaluator_node
from nodes.memory_writer import memory_writer_node
from token_tracker import tracker


SEP  = "=" * 70
DASH = "-" * 70

MAX_LOOP_ATTEMPTS = 3


def should_retry(state: OESAState) -> str:
    """Decide whether to retry the Executor or move to MemoryWriter."""
    eval_passed = state.get("eval_passed", False)
    attempt = state.get("attempt", 1)
    max_attempts = state.get("max_attempts", MAX_LOOP_ATTEMPTS)

    if eval_passed:
        print(f"\n  [GRAPH] Eval passed — moving to memory writer")
        return "memory_writer"

    if attempt >= max_attempts:
        print(f"\n  [GRAPH] Max attempts ({max_attempts}) reached — saving as-is")
        return "memory_writer"

    print(f"\n  [GRAPH] Eval failed — retrying (attempt {attempt + 1}/{max_attempts})")
    return "executor"


def increment_attempt(state: OESAState) -> dict:
    """Increment the attempt counter before retrying executor."""
    return {"attempt": state.get("attempt", 1) + 1}


def build_graph() -> StateGraph:
    """Build the OESA LangGraph pipeline with evaluation feedback loop."""
    graph = StateGraph(OESAState)

    # Add nodes
    graph.add_node("searcher", searcher_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("increment_attempt", increment_attempt)
    graph.add_node("memory_writer", memory_writer_node)

    # Linear flow: searcher → planner → executor → evaluator
    graph.set_entry_point("searcher")
    graph.add_edge("searcher", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "evaluator")

    # Conditional: evaluator → retry executor OR → memory_writer
    graph.add_conditional_edges(
        "evaluator",
        should_retry,
        {
            "executor": "increment_attempt",
            "memory_writer": "memory_writer",
        }
    )
    graph.add_edge("increment_attempt", "executor")
    graph.add_edge("memory_writer", END)

    return graph.compile()


def run_pipeline(task: str):
    """Run the full OESA pipeline for a math domain."""
    print(f"\n{SEP}")
    print(f"  OESA AGENT BUILDER — LangGraph Edition")
    print(f"  Domain: {task}")
    print(SEP)

    tracker.reset()
    app = build_graph()

    initial_state = {
        "task": task,
        "domain": "",
        "relevant_successes": [],
        "known_failures": [],
        "approach_hints": "",
        "workflow_name": "",
        "workflow_description": "",
        "orchestrator_logic": "",
        "agents": [],
        "flow": [],
        "generated_code": "",
        "output_file": "",
        "eval_passed": False,
        "eval_score": 0.0,
        "eval_output": "",
        "eval_expected": "",
        "eval_feedback": "",
        "attempt": 1,
        "max_attempts": MAX_LOOP_ATTEMPTS,
        "error": None,
        "token_usage": None,
    }

    try:
        result = app.invoke(initial_state)

        eval_status = "PASSED" if result.get("eval_passed") else "FAILED"
        eval_score = result.get("eval_score", 0.0)
        attempts = result.get("attempt", 1)
        token_usage = result.get("token_usage") or tracker.totals()

        print(f"\n{SEP}")
        print(f"  DONE")
        print(f"  Solver: {result.get('output_file', '?')}")
        print(f"  Eval: {eval_status} (score={eval_score}, attempts={attempts})")
        print(f"  {tracker.summary_line()}")
        print(SEP)

        return result

    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")
        error_state = {**initial_state, "error": str(e)}
        memory_writer_node(error_state)
        raise


if __name__ == "__main__":
    task = input(
        "Describe the math domain to build a solver for\n"
        "(e.g. 'algebra word problems', 'calculus derivatives', 'geometry proofs'):\n> "
    ).strip()
    if not task:
        task = "algebra word problems"
    run_pipeline(task)
