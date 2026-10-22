"""
OESA LangGraph State — Shared state that flows through the graph.

Each node reads from and writes to this TypedDict.
"""

from typing import TypedDict, Optional


class AgentSpec(TypedDict):
    id: str
    name: str
    role: str
    inputs: list[str]
    outputs: list[str]
    system_prompt: str


class OESAState(TypedDict):
    # Input
    task: str

    # Searcher outputs
    domain: str
    relevant_successes: list[dict]
    known_failures: list[dict]
    approach_hints: str

    # Planner outputs
    workflow_name: str
    workflow_description: str
    orchestrator_logic: str
    agents: list[AgentSpec]
    flow: list[list[str]]

    # Executor outputs
    generated_code: str
    output_file: str

    # Evaluator outputs
    eval_passed: bool
    eval_score: float          # 0.0 to 1.0
    eval_output: str           # what the solver produced
    eval_expected: str         # what the correct answer is
    eval_feedback: str         # error details for retry

    # Loop control
    attempt: int               # current attempt number (1, 2, 3)
    max_attempts: int          # max retries (default 3)

    # Status
    error: Optional[str]

    # Token usage — populated by graph.py after each run
    token_usage: Optional[dict]
