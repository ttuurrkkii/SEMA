# SEMA: Self-Evolving Math Agent

SEMA is a framework that builds a fresh agent pipeline for each math problem type instead of relying on one fixed, hand-crafted solver. It keeps a memory of approaches that worked and approaches that failed, and uses that memory to design better solvers over time.

Built with [LangGraph](https://github.com/langchain-ai/langgraph). Evaluated on GSM8K.

## How it works

```
Searcher → Planner → Executor → Evaluator ──→ MemoryWriter → END
                        ↑           │
                        └── retry ──┘
```

| Node | Role |
|------|------|
| `nodes/searcher.py` | Detects the domain, queries STSC/DBRFPA memory and builds hints for the Planner |
| `nodes/planner.py` | Designs a multi-agent solver pipeline for the task |
| `nodes/executor.py` | Generates the solver code, validates syntax and self-tests it |
| `nodes/evaluator.py` | Runs the solver on test problems and checks the answers |
| `nodes/memory_writer.py` | Records the outcome back into memory |

Memory (`memory/`):
- **STSC** (`stsc.py`), Short-Term Success Cache: recent successful task–solution pairs, by domain.
- **DBRFPA** (`dbrfpa.py`), Domain-Based Reflective Failure Public Archive: past failures, so the Planner avoids repeating them.
- Both search with `all-MiniLM-L6-v2` embeddings and cosine similarity. `seed_golden.py` seeds STSC with known-good solver patterns.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your own API keys
```

By default SEMA uses MiniMax M2 through OpenRouter. Set `LLM_PROVIDER=claude` in `.env` to use Anthropic instead.

## Usage

Build a solver interactively:

```bash
python graph.py
```

Run the GSM8K benchmark:

```bash
python sema_gsm8k_runner.py --n 100                 # GSM8K with memory
python sema_gsm8k_runner.py --n 100 --no-memory     # GSM8K, memory cleared
python sema_gsm8k_runner.py --resume                # resume an interrupted run
python sema_gsm8k_runner.py --results-dir my_dir    # custom output folder
```
