"""
SEMA Framework — GSM8K Benchmark Runner (with token tracking)

Phase 1 — BUILD: SEMA generates one solver per GSM8K category (11 total).
Phase 2 — EVAL:  Each solver runs in-process with a patched OpenAI client
                  so per-problem inference tokens are tracked accurately.

Usage:
  python sema_gsm8k_runner.py              # 100 problems (proportional sample)
  python sema_gsm8k_runner.py --n 200
  python sema_gsm8k_runner.py --resume
  python sema_gsm8k_runner.py --results-dir my_dir
"""

import sys, io, os, json, re, argparse, importlib.util
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

# ---------------------------------------------------------------------------
# GSM8K categories → SEMA task descriptions
# ---------------------------------------------------------------------------

GSM8K_CATEGORIES = [
    {"cat": "basic_arithmetic",    "task": "grade school arithmetic word problem solver (addition, subtraction, multiplication, division)", "example": "Lisa had 12 apples. She ate 4. How many are left?"},
    {"cat": "money_cost",          "task": "grade school money and cost word problem solver (prices, change, totals, budgets)", "example": "A shirt costs $12. A pair of socks costs $3. How much do both cost?"},
    {"cat": "age",                 "task": "grade school age word problem solver (finding ages, age differences, future/past ages)", "example": "Tom is 8. His sister is 3 years older. How old is his sister?"},
    {"cat": "ratios",              "task": "grade school ratio and proportion word problem solver", "example": "The ratio of boys to girls is 2:3. There are 30 students total. How many boys?"},
    {"cat": "percentages",         "task": "grade school percentage word problem solver (percent of, percent change, discounts, tax)", "example": "A $50 jacket is 20% off. What is the sale price?"},
    {"cat": "fractions",           "task": "grade school fraction word problem solver (adding, subtracting, multiplying fractions; word problems)", "example": "Maria ate 1/4 of a pizza. John ate 1/2. How much is left?"},
    {"cat": "rate_speed_distance", "task": "grade school rate, speed, and distance word problem solver (d=r×t problems)", "example": "A car travels at 60 mph for 3 hours. How far does it go?"},
    {"cat": "geometry",            "task": "grade school geometry word problem solver (area, perimeter, volume of simple shapes)", "example": "A rectangle is 8 cm long and 5 cm wide. What is its area?"},
    {"cat": "unit_conversions",    "task": "grade school unit conversion word problem solver (length, weight, time, volume conversions)", "example": "How many minutes are in 3.5 hours?"},
    {"cat": "work_employment",     "task": "grade school work and employment word problem solver (wages, hours, total pay)", "example": "Jane earns $15/hour. She works 6 hours. How much does she earn?"},
    {"cat": "multistep",           "task": "grade school multi-step word problem solver requiring several arithmetic operations", "example": "There are 5 boxes with 8 apples each. 12 apples were eaten. How many remain?"},
]

CAT_NAMES = [c["cat"] for c in GSM8K_CATEGORIES]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_number(text):
    if not text or str(text).startswith("["):
        return ""
    m = re.search(r"####\s*([\-\d,\.]+)", str(text))
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"(?:answer|result|total)\s*(?:is|:|=)\s*\$?([\-\d,\.]+)", str(text), re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\-]?\d+\.?\d*", str(text))
    return nums[-1] if nums else ""


def answers_match(predicted, expected):
    try:
        return abs(float(str(predicted).replace(",", "")) - float(str(expected).replace(",", ""))) < 0.01
    except (ValueError, TypeError):
        return str(predicted).strip() == str(expected).strip()


def load_questions(n):
    path = os.path.join(SCRIPT_DIR, "gsm8k_results", "classified_questions.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"][:n]


# ---------------------------------------------------------------------------
# OpenAI client monkey-patch for in-process token tracking
# ---------------------------------------------------------------------------

_tok_buf = {"prompt": 0, "completion": 0, "calls": 0}
_original_create = None


def _reset_tok():
    _tok_buf["prompt"] = 0
    _tok_buf["completion"] = 0
    _tok_buf["calls"] = 0


def _tok_totals():
    return {
        "prompt_tokens": _tok_buf["prompt"],
        "completion_tokens": _tok_buf["completion"],
        "total_tokens": _tok_buf["prompt"] + _tok_buf["completion"],
        "calls": _tok_buf["calls"],
    }


def _install_patch():
    global _original_create
    try:
        from openai.resources.chat.completions import Completions
        _original_create = Completions.create

        def patched_create(self, *args, **kwargs):
            response = _original_create(self, *args, **kwargs)
            if hasattr(response, "usage") and response.usage:
                _tok_buf["prompt"] += getattr(response.usage, "prompt_tokens", 0) or 0
                _tok_buf["completion"] += getattr(response.usage, "completion_tokens", 0) or 0
                _tok_buf["calls"] += 1
            return response

        Completions.create = patched_create
        print("  [TOKEN PATCH] OpenAI client patched for in-process token tracking.")
    except Exception as e:
        print(f"  [TOKEN PATCH] Warning: could not patch OpenAI client: {e}")


# ---------------------------------------------------------------------------
# Load generated solver module in-process
# ---------------------------------------------------------------------------

_solver_cache = {}


def load_solver_module(solver_path):
    if solver_path in _solver_cache:
        return _solver_cache[solver_path]
    _saved_stdout = sys.stdout
    try:
        spec = importlib.util.spec_from_file_location(
            "solver_mod_" + os.path.splitext(os.path.basename(solver_path))[0],
            solver_path,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _solver_cache[solver_path] = mod
    finally:
        # Detach any new TextIOWrapper the module created so it doesn't
        # close the underlying buffer when garbage-collected
        try:
            if sys.stdout is not _saved_stdout:
                sys.stdout.detach()
        except Exception:
            pass
        sys.stdout = _saved_stdout
    return mod


def run_solver_inprocess(solver_path, problem):
    """Run the solver's Orchestrator in-process and return answer string."""
    try:
        mod = load_solver_module(solver_path)
        orch = mod.Orchestrator()
        return orch.run(problem)
    except Exception as e:
        return f"[ERROR] {e}"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(results_dir):
    path = os.path.join(results_dir, "checkpoint.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "solvers": {},
        "build_tokens": {},
        "results": [],
        "by_category": {},
        "correct": 0,
        "total": 0,
        "total_tokens": 0,
    }


def save_checkpoint(cp, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "checkpoint.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2)


# ---------------------------------------------------------------------------
# Phase 1: Build solvers
# ---------------------------------------------------------------------------

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
        sys.stdout.flush()
    except (ValueError, OSError):
        pass


def build_solvers(cp, results_dir, no_memory=False):
    from graph import build_graph
    from token_tracker import tracker

    if no_memory:
        print("\n  [BUILD] --no-memory: clearing STSC and DBRFPA")
        try:
            from nodes.searcher import stsc, dbrfpa
            import numpy as np
            stsc._store.clear()
            dbrfpa._store.clear()
            stsc._embeddings = np.zeros((0, stsc._embeddings.shape[1] if stsc._embeddings.ndim > 1 else 768))
            dbrfpa._embeddings = np.zeros((0, dbrfpa._embeddings.shape[1] if dbrfpa._embeddings.ndim > 1 else 768))
        except Exception as e:
            print(f"  [BUILD] Warning: could not clear memory: {e}")

    graph = build_graph()
    SEP = "=" * 65

    for cat_info in GSM8K_CATEGORIES:
        cat = cat_info["cat"]
        if cat in cp["solvers"] and cp["solvers"][cat]:
            path = cp["solvers"][cat]
            if os.path.exists(path):
                print(f"  [BUILD] {cat}: already built → {os.path.basename(path)}")
                continue

        print(f"\n{SEP}")
        print(f"  [BUILD] Category: {cat}")
        print(f"  [BUILD] Task: {cat_info['task']}")
        print(SEP)

        tracker.reset()
        initial_state = {
            "task": cat_info["task"],
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
            "eval_expected": cat_info["example"],
            "eval_feedback": "",
            "attempt": 1,
            "max_attempts": 2,
            "error": None,
            "token_usage": None,
        }

        solver_path = None
        try:
            result = graph.invoke(initial_state)
            solver_path = result.get("output_file")
        except Exception as e:
            print(f"  [BUILD] ERROR for {cat}: {e}")

        # In no-memory mode, wipe any writes the memory_writer_node just made
        if no_memory:
            try:
                from nodes.searcher import stsc, dbrfpa
                import numpy as np
                stsc._store.clear()
                dbrfpa._store.clear()
                stsc._embeddings = np.zeros((0, stsc._embeddings.shape[1] if stsc._embeddings.ndim > 1 else 768))
                dbrfpa._embeddings = np.zeros((0, dbrfpa._embeddings.shape[1] if dbrfpa._embeddings.ndim > 1 else 768))
            except Exception:
                pass

        tok = tracker.totals()
        cp["solvers"][cat] = solver_path or ""
        cp["build_tokens"][cat] = tok
        save_checkpoint(cp, results_dir)

        print(f"  [BUILD] {cat}: {solver_path}")
        print(f"  [BUILD] Tokens: {tok['prompt_tokens']}p + {tok['completion_tokens']}c = {tok['total_tokens']} total")
        sys.stdout.flush()

    return cp


# ---------------------------------------------------------------------------
# Phase 2: Evaluate
# ---------------------------------------------------------------------------

def evaluate(cp, questions, results_dir):
    already_done = {r["idx"] for r in cp["results"]}
    SEP = "=" * 65

    # Group questions by category
    by_cat = {c["cat"]: [] for c in GSM8K_CATEGORIES}
    for q in questions:
        cat = q["category"]
        if cat in by_cat:
            by_cat[cat].append(q)

    t0 = datetime.now()

    for cat_info in GSM8K_CATEGORIES:
        cat = cat_info["cat"]
        solver_path = cp["solvers"].get(cat, "")

        if not solver_path or not os.path.exists(solver_path):
            safe_print(f"\n  [EVAL] {cat}: no solver, skipping")
            continue

        probs = by_cat[cat]
        if not probs:
            continue

        safe_print(f"\n{SEP}")
        safe_print(f"  [EVAL] {cat}  ({len(probs)} problems)")
        safe_print(SEP)

        for q in probs:
            idx = q["index"]
            if idx in already_done:
                continue

            _reset_tok()
            t1 = datetime.now()

            try:
                raw = run_solver_inprocess(solver_path, q["question"])
            except Exception as e:
                raw = f"[ERROR] {e}"

            elapsed = (datetime.now() - t1).total_seconds()
            tok = _tok_totals()
            predicted = extract_number(raw)
            correct = answers_match(predicted, q["expected"])

            if cat not in cp["by_category"]:
                cp["by_category"][cat] = {"correct": 0, "total": 0}
            cp["by_category"][cat]["total"] += 1
            if correct:
                cp["by_category"][cat]["correct"] += 1
                cp["correct"] += 1
            cp["total"] += 1
            cp["total_tokens"] += tok["total_tokens"]

            acc_cat = cp["by_category"][cat]["correct"] / cp["by_category"][cat]["total"] * 100
            overall_acc = cp["correct"] / cp["total"] * 100
            status = "OK" if correct else "XX"

            safe_print(
                f"  [{cp['total']:>4}] [{status}] {cat:<22} "
                f"exp={q['expected']:>8s} got={predicted:>8s} "
                f"tok={tok['total_tokens']:>5} ({tok['calls']}calls) {elapsed:.1f}s "
                f"| cat={acc_cat:.1f}% overall={overall_acc:.1f}%"
            )

            cp["results"].append({
                "idx": idx,
                "category": cat,
                "expected": q["expected"],
                "predicted": predicted,
                "correct": correct,
                "tokens": tok["total_tokens"],
                "prompt_tokens": tok["prompt_tokens"],
                "completion_tokens": tok["completion_tokens"],
                "calls": tok["calls"],
                "time": round(elapsed, 1),
            })

            if cp["total"] % 10 == 0:
                save_checkpoint(cp, results_dir)
                elapsed_total = (datetime.now() - t0).total_seconds()
                rate = cp["total"] / elapsed_total if elapsed_total > 0 else 0
                remaining = (len(questions) - cp["total"]) / rate / 60 if rate > 0 else 0
                avg_tok = cp["total_tokens"] / cp["total"]
                safe_print(f"\n  --- {cp['total']}/{len(questions)} | acc={overall_acc:.1f}% | avg_tok={avg_tok:.0f} | ETA {remaining:.1f}min ---\n")

    save_checkpoint(cp, results_dir)
    return cp


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(cp, questions):
    SEP = "=" * 65
    DASH = "-" * 65
    n = len(questions)
    overall_acc = cp["correct"] / cp["total"] * 100 if cp["total"] else 0
    avg_tok = cp["total_tokens"] / cp["total"] if cp["total"] else 0

    # Build token totals
    build_total = sum(v.get("total_tokens", 0) for v in cp["build_tokens"].values())
    build_amort = build_total / cp["total"] if cp["total"] else 0

    print(f"\n{SEP}")
    print(f"  SEMA — GSM8K RESULTS")
    print(SEP)
    print(f"\n  ACCURACY BY CATEGORY:")
    print(f"  {'Category':<24} {'Correct':>7} {'Total':>6} {'Acc':>7}")
    print(f"  {DASH[:46]}")
    for cat_info in GSM8K_CATEGORIES:
        cat = cat_info["cat"]
        s = cp["by_category"].get(cat, {"correct": 0, "total": 0})
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {cat:<24} {s['correct']:>7} {s['total']:>6} {acc:>6.1f}%")
    print(f"  {DASH[:46]}")
    print(f"  {'OVERALL':<24} {cp['correct']:>7} {cp['total']:>6} {overall_acc:>6.1f}%")

    print(f"\n  TOKEN SUMMARY:")
    print(f"  Inference avg/problem : {avg_tok:.0f} tokens")
    print(f"  Build total (11 cats) : {build_total:,} tokens")
    print(f"  Build amortized/prob  : {build_amort:.0f} tokens  (over {cp['total']} problems)")
    print(f"  Combined avg/problem  : {avg_tok + build_amort:.0f} tokens")

    print(f"\n  BUILD TOKENS BY CATEGORY:")
    print(f"  {'Category':<24} {'Build Tok':>10} {'Amort':>8}")
    print(f"  {DASH[:46]}")
    for cat_info in GSM8K_CATEGORIES:
        cat = cat_info["cat"]
        bt = cp["build_tokens"].get(cat, {}).get("total_tokens", 0)
        s = cp["by_category"].get(cat, {"total": 0})
        amort = bt / s["total"] if s["total"] else 0
        print(f"  {cat:<24} {bt:>10,} {amort:>7.0f}/q")

    print(SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SEMA GSM8K runner with token tracking")
    parser.add_argument("--n", type=int, default=100, help="Number of problems (default 100)")
    parser.add_argument("--results-dir", default="sema_gsm8k_results", help="Output directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-memory", action="store_true", help="Clear STSC/DBRFPA before building solvers")
    args = parser.parse_args()

    results_dir = os.path.join(SCRIPT_DIR, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    mem_label = "DISABLED (--no-memory)" if args.no_memory else "ENABLED"
    print(f"\n{'='*65}")
    print(f"  SEMA FRAMEWORK — GSM8K BENCHMARK (with token tracking)")
    print(f"  Problems : {args.n}  |  Categories: {len(GSM8K_CATEGORIES)}")
    print(f"  Memory   : {mem_label}")
    print(f"  Output   : {results_dir}")
    print(f"{'='*65}\n")

    # Install token patch for in-process solver execution
    _install_patch()

    questions = load_questions(args.n)
    print(f"  Loaded {len(questions)} questions from classified_questions.json\n")

    cp = load_checkpoint(results_dir) if args.resume else {
        "solvers": {},
        "build_tokens": {},
        "results": [],
        "by_category": {},
        "correct": 0,
        "total": 0,
        "total_tokens": 0,
    }

    cp = build_solvers(cp, results_dir, no_memory=args.no_memory)
    cp = evaluate(cp, questions, results_dir)
    print_report(cp, questions)


if __name__ == "__main__":
    main()
