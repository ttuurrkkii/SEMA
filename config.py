import os
from dotenv import load_dotenv

load_dotenv()

# --- Provider selection ---
# Set LLM_PROVIDER=minimax to use MiniMax M2 via OpenRouter
# Set LLM_PROVIDER=claude (default) to use Claude Sonnet
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "minimax")

if LLM_PROVIDER == "minimax":
    ANTHROPIC_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = os.getenv("MINIMAX_MODEL", "minimax/minimax-m2")
else:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENROUTER_BASE_URL = None
    MODEL = "claude-sonnet-4-6"

MAX_TOKENS = 4096


def create_llm():
    """Create the LangChain LLM client for the configured provider."""
    if LLM_PROVIDER == "minimax":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            api_key=ANTHROPIC_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://oesa-agent-builder.local",
                "X-Title": "OESA Agent Builder",
            },
        )
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            api_key=ANTHROPIC_API_KEY,
        )
