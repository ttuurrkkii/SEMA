"""
Seed STSC with golden examples — known-good solver patterns.
Run once to bootstrap the memory with high-quality references.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.stsc import STSC

stsc = STSC()

GOLDEN_EXAMPLES = [
    {
        "task": "grade school word problem solver",
        "domain": "agent_builder",
        "approach_used": "Sequential | agents: ['ProblemReader', 'StepByStepSolver', 'AnswerExtractor'] | eval_score=1.0 | CRITICAL: AnswerExtractor must output ONLY the final number, no words or units. Every agent must receive the FULL original problem text, not fragments.",
        "result_summary": "Golden example: 3-agent pipeline. ProblemReader reads and understands the full word problem. StepByStepSolver solves it showing work. AnswerExtractor outputs ONLY the final number.",
    },
    {
        "task": "arithmetic word problem solver with numerical output",
        "domain": "agent_builder",
        "approach_used": "Sequential | agents: ['ProblemAnalyzer', 'MathComputer', 'NumberExtractor'] | eval_score=1.0 | KEY RULES: (1) Pass full problem text to every agent. (2) Last agent outputs ONLY a number. (3) Do NOT fragment or parse the problem into pieces — let the LLM read the full text.",
        "result_summary": "Golden example: The most common failure is agents that parse the problem into fragments and lose context. Solution: pass the FULL problem text through the entire chain. The last agent must be a pure number extractor.",
    },
]

for ex in GOLDEN_EXAMPLES:
    stsc.add(**ex)
    print(f"  Added: {ex['task']}")

print(f"\n  STSC now has {len(stsc)} entries")
