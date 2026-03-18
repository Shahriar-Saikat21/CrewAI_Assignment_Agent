"""
manager.py - The Smart Manager
================================
This is the BRAIN of the whole system. The manager:

  1. CHATS with the user normally (greetings, questions, etc.)
  2. DETECTS when the user wants an assignment created
  3. DELEGATES to sub-agents only when needed (researcher -> writer -> quality checker)
  4. HANDLES the quality loop (if score < 7, sends back to writer)
  5. RETURNS the final assignment to the user

How it decides what to do:
  - User says "hello"                -> Manager just chats back
  - User says "what can you do?"     -> Manager explains its capabilities
  - User says "create an assignment on AI" -> Manager triggers the full pipeline

The pipeline (only when assignment is needed):
  [Researcher] -> [Writer] -> [Quality Checker] -> score >= 7? -> Done!
                     ^               |
                     |         score < 7?
                     └─── Revise (max 2 times) ──┘
"""

from crewai import Crew, Process, LLM
from langchain_openai import ChatOpenAI

from src.agents import create_all_agents
from src.tasks import create_research_task, create_writing_task, create_quality_task
from src.state import AssignmentState
from src.utils import parse_quality_score, save_assignment


class Manager:
    """
    The Manager is the main interface between the user and the sub-agents.
    It uses TWO different LLM connections:
      - chat_llm (ChatOpenAI): for direct chat with user (fast, no agents)
      - crew_llm (CrewAI LLM): for powering sub-agents (uses litellm internally)
    """

    def __init__(self, chat_llm: ChatOpenAI, crew_llm: LLM):
        """
        Initialize the manager with two LLMs.
        - chat_llm: ChatOpenAI instance for direct chat (manager talks to user)
        - crew_llm: CrewAI LLM instance for sub-agents (uses litellm -> openrouter)
        - agents: the 3 sub-agents (created lazily, only when needed)
        - chat_history: keeps track of the conversation for context
        """
        self.chat_llm = chat_llm    # For direct chat (no agents)
        self.crew_llm = crew_llm    # For CrewAI agents (researcher, writer, quality checker)
        self.agents = None           # Sub-agents are NOT created until needed (saves resources)
        self.chat_history = []       # Stores conversation: [{"role": "user/assistant", "content": "..."}]

    def _ensure_agents(self):
        """
        Creates sub-agents only when needed (lazy initialization).
        Called once before the first assignment creation.
        After that, the agents are reused.
        """
        if self.agents is None:
            print("\n  [Manager] Initializing sub-agents...")
            self.agents = create_all_agents(self.crew_llm)  # Uses CrewAI LLM for agents

    def _classify_intent(self, user_message: str) -> str:
        """
        Uses the LLM to decide: is the user asking for an assignment, or just chatting?

        Returns:
          - "assignment" if the user wants an assignment/essay/report created
          - "chat" if the user is greeting or asking about this tool's capabilities
          - "off_topic" if the user is asking about something outside our scope

        The LLM reads the message and the recent chat history to understand context.
        For example: "yes do it" after discussing a topic = assignment (from context).
        """
        # Build recent chat context (last 6 messages) so the LLM understands conversation flow
        recent_context = ""
        if self.chat_history:
            recent = self.chat_history[-6:]  # Last 6 messages for context
            recent_context = "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in recent]
            )
            recent_context = f"\nRecent conversation:\n{recent_context}\n"

        # Ask the LLM to classify the intent
        classification_prompt = f"""{recent_context}
Current user message: "{user_message}"

You are an academic assignment creation assistant. Your ONLY purpose is to create assignments on topics.

Classify the user's message into EXACTLY one of these categories:
- ASSIGNMENT (user wants to create an assignment, essay, report, or academic paper on a topic)
- CHAT (user is greeting, saying thanks, asking what you can do, or talking about assignments in general)
- OFF_TOPIC (user is asking about something unrelated like weather, news, coding help, personal advice, etc.)

Reply with EXACTLY one word: ASSIGNMENT, CHAT, or OFF_TOPIC"""

        response = self.chat_llm.invoke(classification_prompt)  # Ask the ChatOpenAI LLM
        result = response.content.strip().upper()

        # Return intent based on LLM's response
        if "ASSIGNMENT" in result:
            return "assignment"
        if "OFF_TOPIC" in result:
            return "off_topic"
        return "chat"

    def _extract_subject(self, user_message: str) -> str:
        """
        Uses the LLM to extract the assignment subject/topic from the user's message.

        Example: "create an assignment on machine learning" -> "Machine Learning"
        Example: "write about climate change effects" -> "Climate Change Effects"

        Also considers chat history in case user said the topic earlier.
        """
        recent_context = ""
        if self.chat_history:
            recent = self.chat_history[-6:]
            recent_context = "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in recent]
            )
            recent_context = f"\nRecent conversation:\n{recent_context}\n"

        extract_prompt = f"""{recent_context}
Current user message: "{user_message}"

Extract the subject/topic that the user wants an assignment about.
Reply with ONLY the topic name, nothing else.
Example: "Machine Learning" or "Climate Change Effects on Agriculture"."""

        response = self.chat_llm.invoke(extract_prompt)
        return response.content.strip().strip('"').strip("'")  # Clean up quotes if any

    def _off_topic_response(self) -> str:
        """
        Returns a polite rejection when the user asks something outside our scope.
        No LLM call needed - just a static message.
        """
        return (
            "Sorry, I can only help with creating academic assignments. "
            "Tell me a subject/topic and I'll create a detailed assignment for you!\n\n"
            "Example: \"Create an assignment on machine learning\""
        )

    def _chat_response(self, user_message: str) -> str:
        """
        Handles on-topic chat (greetings, questions about capabilities, etc.).
        Uses the LLM directly (no sub-agents needed).
        The manager responds as a friendly academic assistant, but stays on scope.
        """
        # Build the full chat history for context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly academic assignment assistant. "
                    "Your ONLY purpose is to create detailed assignments on topics the user provides. "
                    "You can greet the user, explain what you do, and ask for a topic. "
                    "Do NOT answer questions unrelated to assignments (like weather, news, coding, etc.). "
                    "Keep responses short and encourage the user to give you a topic."
                ),
            }
        ]
        # Add recent chat history (last 10 messages) for context
        for msg in self.chat_history[-10:]:
            messages.append(msg)
        # Add the current user message
        messages.append({"role": "user", "content": user_message})

        response = self.chat_llm.invoke(messages)  # Direct ChatOpenAI call (no agents, no crew)
        return response.content

    def _run_assignment_pipeline(self, subject: str) -> str:
        """
        The full assignment creation pipeline. Only called when user wants an assignment.
        This is where the 3 sub-agents do their work:

        Step 1: Researcher gathers information
        Step 2: Writer creates the assignment
        Step 3: Quality Checker reviews and scores (1-10)
        Step 4: If score < 7 and revisions < 2, go back to Step 2 with feedback
        Step 5: Save to file and return the final assignment
        """
        self._ensure_agents()  # Create sub-agents if not already created

        state = AssignmentState(subject=subject)  # Shared state for this assignment

        # ── STEP 1: Research ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  [Researcher] Researching '{subject}'...")
        print(f"{'='*60}\n")

        research_task = create_research_task(subject, self.agents["researcher"])
        research_crew = Crew(
            agents=[self.agents["researcher"]],
            tasks=[research_task],
            process=Process.sequential,
            verbose=True,
        )
        result = research_crew.kickoff()
        state.research = result.raw  # Save research to state

        # ── STEP 2-4: Write -> Quality Check -> Revise Loop ────────────
        while True:
            # STEP 2: Write the assignment
            print(f"\n{'='*60}")
            print(f"  [Writer] Writing assignment (Revision #{state.revision_count})...")
            print(f"{'='*60}\n")

            writing_task = create_writing_task(
                subject,
                state.research,                     # Research from Step 1
                self.agents["writer"],
                feedback=state.quality_feedback,     # Empty on first write, filled on revisions
            )
            writing_crew = Crew(
                agents=[self.agents["writer"]],
                tasks=[writing_task],
                process=Process.sequential,
                verbose=True,
            )
            result = writing_crew.kickoff()
            state.draft = result.raw  # Save draft to state

            # STEP 3: Quality check
            print(f"\n{'='*60}")
            print(f"  [Quality Checker] Reviewing assignment...")
            print(f"{'='*60}\n")

            quality_task = create_quality_task(subject, state.draft, self.agents["quality_checker"])
            quality_crew = Crew(
                agents=[self.agents["quality_checker"]],
                tasks=[quality_task],
                process=Process.sequential,
                verbose=True,
            )
            result = quality_crew.kickoff()
            state.quality_feedback = result.raw
            state.quality_score = parse_quality_score(result.raw)  # Extract score from text

            print(f"\n>> Quality Score: {state.quality_score}/10")

            # STEP 4: Decide - approve or revise?
            if state.quality_score >= 7.0:
                print(">> APPROVED! Assignment passed quality check.")
                break  # Exit the loop - assignment is good
            elif state.revision_count >= state.max_revisions:
                print(">> Max revisions reached. Accepting current draft.")
                break  # Exit the loop - can't revise anymore
            else:
                state.revision_count += 1
                print(f">> Score too low. Sending back for revision #{state.revision_count}...")
                # Loop continues -> goes back to Step 2 with feedback

        # ── STEP 5: Save and return ─────────────────────────────────────
        filename = save_assignment(subject, state.draft)

        print(f"\n{'='*60}")
        print(f"  DONE! Assignment saved to: {filename}")
        print(f"  Quality Score: {state.quality_score}/10")
        print(f"  Revisions: {state.revision_count}")
        print(f"{'='*60}\n")

        return state.draft  # Return the final assignment text

    def handle_message(self, user_message: str) -> str:
        """
        Main entry point - called for every user message.
        This is the method that decides what to do:

        1. Save the user message to chat history
        2. Classify intent: is it "assignment" or "chat"?
        3. If assignment -> extract subject -> run pipeline -> return assignment
        4. If chat -> respond normally using LLM
        5. Save the response to chat history
        """
        # Save user message to history (so future messages have context)
        self.chat_history.append({"role": "user", "content": user_message})

        # Classify: does the user want an assignment or just chatting?
        intent = self._classify_intent(user_message)
        print(f"\n  [Manager] Detected intent: {intent}")

        if intent == "assignment":
            # Extract the subject from the user's message
            subject = self._extract_subject(user_message)
            print(f"  [Manager] Extracted subject: '{subject}'")
            print(f"  [Manager] Starting assignment pipeline...\n")

            # Run the full pipeline: research -> write -> quality check -> (revise) -> done
            response = self._run_assignment_pipeline(subject)

            # Save a short note to chat history (not the full assignment - too long)
            self.chat_history.append({
                "role": "assistant",
                "content": f"I've created a detailed assignment on '{subject}'. The file has been saved.",
            })
        elif intent == "off_topic":
            # User asked something outside our scope (weather, news, etc.)
            response = self._off_topic_response()

            # Save response to history
            self.chat_history.append({"role": "assistant", "content": response})
        else:
            # On-topic chat (greetings, asking about capabilities, etc.)
            response = self._chat_response(user_message)

            # Save assistant response to history
            self.chat_history.append({"role": "assistant", "content": response})

        return response
