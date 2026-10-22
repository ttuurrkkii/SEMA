"""
Evaluator Node — LangGraph version

Runs the generated solver on a sample problem to verify it works.
Returns pass/fail with feedback for retry.
"""

import os
import re
import sys
import subprocess
from state import OESAState


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sample test problems per domain with known answers
DOMAIN_TESTS = {
    "arithmetic": ("If John has 5 apples and buys 3 more, how many apples does he have?", "8"),
    "algebra": ("If 2x + 3 = 11, what is x?", "4"),
    "calculus": ("What is the derivative of x^2?", "2x"),
    "geometry": ("What is the area of a rectangle with length 6 and width 4?", "24"),
    "trigonometry": ("What is sin(30 degrees)?", "0.5"),
    "statistics": ("What is the mean of 2, 4, 6, 8, 10?", "6"),
    "percentage": ("What is 20% of 150?", "30"),
    "fraction": ("What is 1/4 + 1/2?", "0.75"),
    "ratio": ("If the ratio of boys to girls is 2:3 and there are 10 boys, how many girls are there?", "15"),
    "money": ("A shirt costs $25. If you buy 3 shirts, how much do you pay?", "75"),
    "age": ("Tom is 10 years old. His sister is twice his age. How old is his sister?", "20"),
    "rate": ("A car travels at 60 mph for 2 hours. How far does it go?", "120"),
    "speed": ("A car travels at 60 mph for 2 hours. How far does it go?", "120"),
    "distance": ("A car travels at 60 mph for 2 hours. How far does it go?", "120"),
    "work": ("If you earn $15 per hour and work 8 hours, how much do you earn?", "120"),
    "employ": ("If you earn $15 per hour and work 8 hours, how much do you earn?", "120"),
    "wage": ("If you earn $15 per hour and work 8 hours, how much do you earn?", "120"),
    "unit": ("If 1 meter = 100 centimeters, how many centimeters are in 3.5 meters?", "350"),
    "conversion": ("If 1 meter = 100 centimeters, how many centimeters are in 3.5 meters?", "350"),
    "multi": ("A store sells apples for $2 each and oranges for $3 each. If you buy 4 apples and 2 oranges, how much do you spend?", "14"),
}


def _get_test_problem(task: str, domain: str) -> tuple[str, str]:
    """Get a sample problem and expected answer for this domain."""
    search = (task + " " + domain).lower()
    for key, (problem, answer) in DOMAIN_TESTS.items():
        if key in search:
            return problem, answer
    # Default
    return "If you have 12 cookies and eat 5, how many are left?", "7"


def _run_solver(solver_path: str, problem: str, timeout: int = 90) -> str:
    """Run the generated solver and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, solver_path],
            input=problem + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace',
            cwd=SCRIPT_DIR,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        output = result.stdout.strip()
        if not output and result.stderr:
            return f"[ERROR] {result.stderr[:500]}"
        return output
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR] {e}"


def _extract_number(text: str) -> str:
    """Extract the final numerical answer from solver output."""
    if not text or text.startswith("[ERROR]") or text.startswith("[TIMEOUT]"):
        return ""

    # Try common patterns
    for pattern in [
        r'####\s*([\-\d,\.]+)',
        r'(?:answer|result|total)\s*(?:is|:|=)\s*\$?([\-\d,\.]+)',
        r'=\s*\$?([\-\d,\.]+)\s*$',
        r'\$\s*([\-\d,\.]+)',
    ]:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "").strip()

    # Last number in output
    numbers = re.findall(r'([\-]?\d+\.?\d*)', text)
    if numbers:
        return numbers[-1]

    return ""


def _check_answer(predicted: str, expected: str) -> bool:
    """Check if the predicted answer matches expected."""
    if not predicted:
        return False
    try:
        return abs(float(predicted) - float(expected)) < 0.01
    except (ValueError, TypeError):
        return predicted.strip() == expected.strip()


def evaluator_node(state: OESAState) -> dict:
    """LangGraph node: run the generated solver and evaluate output."""
    output_file = state.get("output_file", "")
    task = state.get("task", "")
    domain = state.get("domain", "")
    attempt = state.get("attempt", 1)

    print(f"\n  [EVALUATOR] Testing generated solver (attempt {attempt})...")

    if not output_file or not os.path.exists(output_file):
        print(f"  [EVALUATOR] No solver file found")
        return {
            "eval_passed": False,
            "eval_score": 0.0,
            "eval_output": "",
            "eval_expected": "",
            "eval_feedback": "No solver file was generated.",
        }

    # Get test problem
    problem, expected = _get_test_problem(task, domain)
    print(f"  [EVALUATOR] Test: {problem}")
    print(f"  [EVALUATOR] Expected: {expected}")

    # Run solver
    raw_output = _run_solver(output_file, problem)
    predicted = _extract_number(raw_output)
    print(f"  [EVALUATOR] Got: {predicted}")

    # Check for errors
    if raw_output.startswith("[TIMEOUT]"):
        feedback = "Solver timed out after 90 seconds. The code may have an infinite loop or is waiting for input incorrectly."
        print(f"  [EVALUATOR] TIMEOUT")
        return {
            "eval_passed": False,
            "eval_score": 0.0,
            "eval_output": raw_output,
            "eval_expected": expected,
            "eval_feedback": feedback,
        }

    if raw_output.startswith("[ERROR]"):
        feedback = f"Solver crashed with error: {raw_output[8:]}"
        print(f"  [EVALUATOR] ERROR: {raw_output[:200]}")
        return {
            "eval_passed": False,
            "eval_score": 0.0,
            "eval_output": raw_output,
            "eval_expected": expected,
            "eval_feedback": feedback,
        }

    if not predicted:
        feedback = f"Solver produced output but no numerical answer could be extracted. Raw output: {raw_output[:300]}"
        print(f"  [EVALUATOR] NO ANSWER EXTRACTED")
        return {
            "eval_passed": False,
            "eval_score": 0.2,
            "eval_output": raw_output[:500],
            "eval_expected": expected,
            "eval_feedback": feedback,
        }

    # Check correctness
    correct = _check_answer(predicted, expected)

    if correct:
        print(f"  [EVALUATOR] CORRECT!")
        return {
            "eval_passed": True,
            "eval_score": 1.0,
            "eval_output": predicted,
            "eval_expected": expected,
            "eval_feedback": "",
        }
    else:
        feedback = f"Solver returned {predicted} but expected {expected}. Test problem was: '{problem}'. The solver's reasoning may be wrong or it may be extracting the wrong number from the output."
        print(f"  [EVALUATOR] WRONG (got {predicted}, expected {expected})")
        return {
            "eval_passed": False,
            "eval_score": 0.5,
            "eval_output": predicted,
            "eval_expected": expected,
            "eval_feedback": feedback,
        }
