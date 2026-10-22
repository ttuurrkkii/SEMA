"""
Planner Node — LangGraph version

Designs a multi-agent workflow for the given math domain.
Outputs agent specs and flow edges.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from config import create_llm
from state import OESAState
from token_tracker import tracker


llm = create_llm()


WORKFLOW_SYSTEM = """You are a math agent workflow architect.

Given a description of a math problem domain, design a lean 2-agent pipeline
specialized for solving that type of math problem efficiently.

Output ONLY valid JSON (no markdown fences, no explanation):
{
  "name": "PascalCaseWorkflowName",
  "description": "One sentence: what this pipeline solves",
  "orchestrator_logic": "Sequential",
  "agents": [
    {
      "id": "snake_case_id",
      "name": "PascalCaseName",
      "role": "What this specific agent does",
      "inputs": ["key1", "key2"],
      "outputs": ["key3"],
      "system_prompt": "You are a math agent specialized in ... Your job is to ..."
    }
  ],
  "flow": [["input","first_agent_id"],["first_agent_id","second_agent_id"],["second_agent_id","output"]]
}

Rules:
- EXACTLY 2 agents: a Solver and a Verifier
- Agent 1 — Solver:
    - id: "solver", name: "MathSolver" (adapt to domain, e.g. "AdditionSolver")
    - role: solve the problem with full step-by-step chain-of-thought reasoning
    - system_prompt: specialized for the exact math domain; instructs to show all working steps and end with "#### <number>"
- Agent 2 — Verifier:
    - id: "verifier", name: "AnswerVerifier"
    - role: re-read the original problem and the solution, check the answer is correct, output ONLY the final number
    - system_prompt MUST say: "You are a math verifier. Read the original problem and the solution carefully. Check if the answer is correct. Output ONLY the final numerical answer as a single number. No words, no units, no explanation. Just the number."
- flow: [["input","solver"],["solver","verifier"],["verifier","output"]]
- name must end in "Solver" or "Pipeline"
- CRITICAL: Both agents MUST receive the original problem text
- CRITICAL: The Solver's system_prompt must be highly specific to the math domain (mention domain, operation types, key strategies)
- Output ONLY the JSON object, nothing else"""


FALLBACK_WORKFLOW = {
    "name": "MathSolver",
    "description": "A general math problem solver with verification",
    "orchestrator_logic": "Sequential",
    "agents": [
        {
            "id": "solver", "name": "MathSolver",
            "role": "Solve the math problem step by step with chain-of-thought reasoning",
            "inputs": ["problem"], "outputs": ["solution", "original_problem"],
            "system_prompt": "You are a math solver. Read the full problem carefully and solve it step by step, showing all working. End your response with: #### <number>"
        },
        {
            "id": "verifier", "name": "AnswerVerifier",
            "role": "Verify the solution and output only the final number",
            "inputs": ["solution", "original_problem"], "outputs": ["answer"],
            "system_prompt": "You are a math verifier. Read the original problem and the solution carefully. Check if the answer is correct. Output ONLY the final numerical answer as a single number. No words, no units, no explanation. Just the number."
        },
    ],
    "flow": [["input","solver"],["solver","verifier"],["verifier","output"]],
}


def parse_json(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return FALLBACK_WORKFLOW


def planner_node(state: OESAState) -> dict:
    """LangGraph node: design the agent workflow."""
    task = state["task"]
    domain = state.get("domain", "math")
    hints = state.get("approach_hints", "")
    successes = state.get("relevant_successes", [])
    failures = state.get("known_failures", [])

    print("\n  [PLANNER] Designing agent workflow...")

    # Build prompt
    lines = [
        f"Math domain to build a solver for: {task}",
        f"Detected domain: {domain}",
        "",
        "Context from memory and web search:",
        hints[:500],
    ]
    if successes:
        lines += ["", "Past successful pipelines:"]
        for s in successes:
            lines.append(f"  - {s['approach_used']}")
    if failures:
        lines += ["", "Known failures to avoid:"]
        for f in failures:
            lines.append(f"  - {f['failure_reason']}")
    lines += ["", "Design the agent workflow JSON now."]

    prompt = "\n".join(lines)

    response = llm.invoke([
        SystemMessage(content=WORKFLOW_SYSTEM),
        HumanMessage(content=prompt),
    ])
    tracker.record("planner", response)

    data = parse_json(response.content)

    # Retry once if malformed
    if "agents" not in data or not data["agents"]:
        print("  [PLANNER] JSON malformed, retrying...")
        response = llm.invoke([
            SystemMessage(content=WORKFLOW_SYSTEM),
            HumanMessage(content=prompt),
        ])
        tracker.record("planner", response)
        data = parse_json(response.content)

    agents = [
        {
            "id": a.get("id", f"agent_{i}"),
            "name": a.get("name", f"Agent{i}"),
            "role": a.get("role", ""),
            "inputs": a.get("inputs", ["input"]),
            "outputs": a.get("outputs", ["output"]),
            "system_prompt": a.get("system_prompt", "You are a math agent."),
        }
        for i, a in enumerate(data.get("agents", []))
    ]

    workflow_name = data.get("name", "MathSolver")

    print(f"  [PLANNER] Workflow: {workflow_name}")
    print(f"  [PLANNER] Logic: {data.get('orchestrator_logic', 'Sequential')}")
    for a in agents:
        print(f"    [{a['id']}] {a['name']} — {a['role']}")

    return {
        "workflow_name": workflow_name,
        "workflow_description": data.get("description", ""),
        "orchestrator_logic": data.get("orchestrator_logic", "Sequential"),
        "agents": agents,
        "flow": data.get("flow", []),
    }
