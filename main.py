"""
main.py - Entry Point
=====================
This is the starting point of the application.
Runs a chat loop where the user talks to the Manager agent.

The Manager is smart:
  - If you just chat ("hello", "what can you do?") -> it responds normally
  - If you ask for an assignment ("create assignment on AI") -> it triggers sub-agents

Type 'quit' or 'exit' to stop.
"""

import os
from dotenv import load_dotenv  # load_dotenv reads the .env file and sets environment variables

# Load environment variables from .env file (must be called before accessing OPENROUTER_API_KEY)
load_dotenv()

from src.config import get_chat_llm, get_crew_llm  # Two LLM creators: one for chat, one for agents
from src.manager import Manager                      # The smart manager that handles chat + delegation


def main():
    # Check if API key exists
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set. Add it to your .env file.")
        return

    # Create TWO LLM instances:
    # 1. chat_llm: ChatOpenAI for direct chat (manager talks to user) - uses langchain
    # 2. crew_llm: CrewAI LLM for sub-agents (researcher, writer, etc.) - uses litellm
    chat_llm = get_chat_llm()
    crew_llm = get_crew_llm()

    # Create the manager with both LLMs
    manager = Manager(chat_llm=chat_llm, crew_llm=crew_llm)

    print("=" * 60)
    print("  Assignment Agent")
    print("  Chat normally or ask me to create an assignment!")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    # Chat loop - keeps running until user types 'quit'
    while True:
        user_input = input("\nYou: ").strip()

        # Exit conditions
        if not user_input:
            continue  # Skip empty input
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        # Send message to manager and get response
        response = manager.handle_message(user_input)

        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
