"""
config.py - LLM Configuration
==============================
Sets up TWO types of LLM connections:

1. ChatOpenAI (for manager's direct chat) - uses langchain, talks to OpenRouter directly
2. CrewAI LLM (for sub-agents) - uses litellm under the hood, needs "openrouter/" prefix

Why two? Because:
  - ChatOpenAI: just needs a URL + key, doesn't care about model name format
  - CrewAI agents: use litellm internally, which needs "openrouter/" prefix to know the provider

To change the model, update FREE_MODEL below.
Free model alternatives: "meta-llama/llama-3.1-8b-instruct:free", "google/gemma-2-9b-it:free"
"""

import os
from crewai import LLM                    # CrewAI's native LLM class (uses litellm internally)
from langchain_openai import ChatOpenAI    # OpenAI-compatible chat client (for direct chat)

# The free model name on OpenRouter
FREE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def get_chat_llm(model: str = FREE_MODEL) -> ChatOpenAI:
    """
    Creates an LLM for the MANAGER's direct chat (non-agent conversations).
    Uses ChatOpenAI from LangChain which talks to OpenRouter directly.
    No prefix needed - just sends the model name to the OpenRouter URL.
    """
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",   # Required by OpenRouter
            "X-Title": "AssignmentAgent",         # Optional - shows in OpenRouter dashboard
        },
        temperature=0.7,
        max_tokens=4096,
    )


def get_crew_llm(model: str = FREE_MODEL) -> LLM:
    """
    Creates an LLM for CrewAI AGENTS (researcher, writer, quality checker).
    Uses CrewAI's native LLM class which uses litellm internally.

    IMPORTANT: litellm needs "openrouter/" prefix to know the provider.
    So "nvidia/nemotron-3-super-120b-a12b:free" becomes
       "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    """
    return LLM(
        model=f"openrouter/{model}",                      # "openrouter/" prefix tells litellm the provider
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0.7,
        max_tokens=4096,
    )
