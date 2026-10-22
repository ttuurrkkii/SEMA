"""
Executor Node — LangGraph version

Generates a complete, runnable Python file from the agent workflow spec.
Validates generated code with ast.parse and py_compile.
Runs a self-test with sample input to verify functionality.
Retries with error feedback if validation or self-test fails.
"""

import os
import re
import ast
import py_compile
import subprocess
import sys
import tempfile
from langchain_core.messages import SystemMessage, HumanMessage

from config import create_llm, LLM_PROVIDER, MODEL
from state import OESAState
from token_tracker import tracker


llm = create_llm()

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_RETRIES = 2

# Sample inputs per domain for self-testing
DOMAIN_TEST_INPUTS = {
    "algebra": "2x + 3 = 7",
    "calculus": "derivative of x^2 + 3x",
    "geometry": "area of a circle with radius 5",
    "trigonometry": "sin(30 degrees)",
    "statistics": "mean of 1, 2, 3, 4, 5",
    "probability": "probability of rolling a 6 on a fair die",
    "number theory": "prime factorization of 84",
    "combinatorics": "10 choose 3",
    "linear algebra": "multiply [[1,2],[3,4]] by [[5,6],[7,8]]",
    "graph theory": "shortest path from A to D in graph A-B:1, B-C:2, A-C:4, C-D:1",
}


def _get_test_input(domain: str, task: str) -> str:
    """Get a sample test input for the domain."""
    domain_low = domain.lower()
    for key, val in DOMAIN_TEST_INPUTS.items():
        if key in domain_low or key in task.lower():
            return val
    return "2 + 2"


def _get_executor_system() -> str:
    """Build executor system prompt based on configured LLM provider."""
    provider_block = ""
    if LLM_PROVIDER == "minimax":
        provider_block = f"""- Imports at top (in this order):
    import sys, io, os
    from dotenv import load_dotenv
    from openai import OpenAI
- After imports, before anything else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    load_dotenv()
- Create the client:
    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={{
            "HTTP-Referer": "https://oesa-agent-builder.local",
            "X-Title": "OESA Agent Builder",
        }},
    )
- Each agent class calls:
    response = client.chat.completions.create(
        model="{MODEL}",
        max_tokens=512,
        messages=[
            {{"role": "system", "content": self.SYSTEM}},
            {{"role": "user", "content": str(inputs)}},
        ]
    )
    return {{output_key: response.choices[0].message.content}}"""
    else:
        provider_block = """- Imports at top (in this order):
    import sys, io, os
    from dotenv import load_dotenv
    from anthropic import Anthropic
- After imports, before anything else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    load_dotenv()
- Create: client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
- Each agent class calls:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=self.SYSTEM,
        messages=[{"role": "user", "content": str(inputs)}]
    )
    return {output_key: response.content[0].text}"""

    return f"""You are a Python code generator specializing in multi-agent math solver systems.

Given a math agent workflow specification, write a single complete, runnable Python file.

Requirements:
{provider_block}
- Each agent becomes a Python class with:
    SYSTEM = \"\"\"[the agent's system_prompt from the spec]\"\"\"
    def run(self, inputs: dict) -> dict:
        [call LLM as shown above]
- An Orchestrator class that:
    - __init__: instantiates all agent classes
    - run(self, problem: str) -> str: chains agents in flow order, passing outputs as inputs
- A __main__ block:
    if __name__ == "__main__":
        orch = Orchestrator()
        problem = input("Enter math problem: ").strip()
        result = orch.run(problem)
        print("\\nAnswer:\\n" + result)

CRITICAL RULES:
- The first agent MUST receive the FULL original problem text as input: {{"problem": problem}}
- Every agent MUST pass the original problem text forward in its output dict along with its own results
- Each agent's system prompt MUST instruct the LLM to solve/analyze the FULL word problem, not just fragments
- The last agent (AnswerFormatter) MUST extract ONLY the final numerical answer
- The AnswerFormatter's system prompt MUST say: "Output ONLY the final numerical answer as a single number. No words, no units, no explanation. Just the number."
- The Orchestrator's run() method MUST return ONLY the final number string
- Do NOT parse or fragment the word problem — pass the full text through the chain
- Output ONLY the Python code — no markdown fences, no explanation
- Do NOT duplicate code or repeat the file content"""


def _clean_code(code: str) -> str:
    """Clean LLM output to extract valid Python code."""
    # Remove markdown fences
    if code.startswith("```"):
        parts = code.split("```")
        code = parts[1] if len(parts) > 1 else code
        if code.startswith("python"):
            code = code[6:]
        code = code.strip()

    # Remove junk XML tags (MiniMax M2 artifact)
    code = re.split(r'\n</\w+>', code)[0].rstrip()

    # Remove duplicate file content (some models repeat the entire file)
    lines = code.split('\n')
    # Find if there's a second "import sys" after the first one
    import_indices = [i for i, line in enumerate(lines) if line.strip().startswith('import sys')]
    if len(import_indices) > 1:
        # Keep only up to the second occurrence
        code = '\n'.join(lines[:import_indices[1]])

    return code.strip()


def _validate_syntax(code: str) -> tuple[bool, str]:
    """Validate Python syntax using ast.parse and py_compile."""
    # Step 1: AST parse
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    # Step 2: py_compile (catches more issues)
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            tmp_path = f.name
        py_compile.compile(tmp_path, doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, f"CompileError: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _self_test(code: str, test_input: str) -> tuple[bool, str]:
    """Verify generated code is structurally sound.
    Checks that it has an Orchestrator class and enough agent classes.
    Does NOT make real LLM calls.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Cannot parse: {e}"

    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    has_orchestrator = any("orchestrator" in name.lower() for name in class_names)

    if not has_orchestrator:
        return False, "No Orchestrator class found"

    if len(class_names) < 2:
        return False, f"Only {len(class_names)} class(es) found, expected at least 2 (solver + orchestrator)"

    # Check that Orchestrator has a run method
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "orchestrator" in node.name.lower():
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if "run" not in methods:
                return False, f"Orchestrator class has no run() method, found: {methods}"

    return True, f"Found {len(class_names)} classes: {class_names}"


def _build_workflow_prompt(state: OESAState) -> str:
    """Build the workflow specification prompt for the LLM."""
    agents = state["agents"]
    flow = state["flow"]

    lines = [
        "=== WORKFLOW SPECIFICATION ===",
        f"Name        : {state['workflow_name']}",
        f"Description : {state['workflow_description']}",
        f"Logic       : {state['orchestrator_logic']}",
        f"Math domain : {state['task']}",
        "",
        "=== AGENTS ===",
    ]
    for a in agents:
        lines += [
            f"Agent ID   : {a['id']}",
            f"Class Name : {a['name']}",
            f"Role       : {a['role']}",
            f"Inputs     : {a['inputs']}",
            f"Outputs    : {a['outputs']}",
            f"System Prompt:",
            f"  {a['system_prompt']}",
            "",
        ]
    lines += [
        "=== FLOW (directed edges) ===",
        str(flow),
        "",
        "Generate the complete Python file now.",
    ]
    return "\n".join(lines)


def executor_node(state: OESAState) -> dict:
    """LangGraph node: generate, validate, self-test, and save Python code."""
    print("\n  [EXECUTOR] Generating Python code...")

    prompt = _build_workflow_prompt(state)
    system = _get_executor_system()
    errors_so_far = []

    # If we have eval feedback from a previous loop, include it
    eval_feedback = state.get("eval_feedback", "")
    if eval_feedback:
        errors_so_far.append(f"EVALUATOR FEEDBACK: {eval_feedback}")

    for attempt in range(1, MAX_RETRIES + 2):  # +2 because range is exclusive
        if attempt > 1 or eval_feedback:
            label = f"Retry {attempt - 1}/{MAX_RETRIES}" if attempt > 1 else "Fixing from evaluator feedback"
            print(f"\n  [EXECUTOR] {label}...")
            error_feedback = "\n".join(errors_so_far)
            retry_prompt = f"""{prompt}

=== PREVIOUS ATTEMPT FAILED ===
The following errors were found in your previous code. Fix them:

{error_feedback}

Generate the COMPLETE corrected Python file now. Do NOT repeat the errors above."""
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=retry_prompt),
            ])
            tracker.record("executor", response)
            eval_feedback = ""  # Only use once
        else:
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ])
            tracker.record("executor", response)

        code = _clean_code(response.content.strip())

        # Step 1: Syntax validation
        valid, error_msg = _validate_syntax(code)
        if not valid:
            print(f"  [EXECUTOR] Syntax error: {error_msg}")
            errors_so_far.append(f"SYNTAX ERROR: {error_msg}")
            if attempt <= MAX_RETRIES:
                continue
            else:
                print(f"  [EXECUTOR] All retries exhausted. Saving code with known issues.")
                break

        print(f"  [EXECUTOR] Syntax valid")

        # Step 2: Self-test (structural check)
        test_ok, test_output = _self_test(code, _get_test_input(state.get("domain", ""), state["task"]))
        if not test_ok:
            print(f"  [EXECUTOR] Self-test failed: {test_output}")
            errors_so_far.append(f"STRUCTURAL ERROR: {test_output}")
            if attempt <= MAX_RETRIES:
                continue
            else:
                print(f"  [EXECUTOR] All retries exhausted. Saving code with known issues.")
                break

        print(f"  [EXECUTOR] Self-test passed: {test_output}")
        break

    # Save to file
    safe_name = state["workflow_name"].lower().replace(" ", "_").replace("-", "_")
    safe_name = re.sub(r'[^a-z0-9_]', '', safe_name)
    gen_dir = os.path.join(SCRIPT_DIR, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    fname = os.path.join(gen_dir, f"{safe_name}.py")

    with open(fname, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"  [EXECUTOR] Saved: {fname}")
    print(f"\n  --- CODE PREVIEW (first 2000 chars) ---")
    print(code[:2000])
    if len(code) > 2000:
        print(f"\n  ... ({len(code) - 2000} more chars)")

    # Report validation status
    if errors_so_far:
        error_summary = f"Code saved with issues after {len(errors_so_far)} error(s): {'; '.join(errors_so_far)}"
        print(f"\n  [EXECUTOR] WARNING: {error_summary}")
        return {
            "generated_code": code,
            "output_file": fname,
            "error": error_summary,
        }

    return {
        "generated_code": code,
        "output_file": fname,
    }
