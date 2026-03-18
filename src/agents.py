"""
agents.py - Agent Definitions
==============================
Defines 3 sub-agents that the manager calls ONLY when an assignment is needed.
These agents are NOT always active - they only work when delegated to.

These agents use CrewAI's native LLM class (not ChatOpenAI) because
CrewAI internally uses litellm to talk to the model.

Sub-Agents:
  1. Researcher      - Gathers information on the topic
  2. Writer          - Writes the assignment using research
  3. Quality Checker - Reviews and scores the assignment (1-10)
"""

from crewai import Agent, LLM


def create_researcher(llm: LLM) -> Agent:
    """
    Researcher Agent - called by manager when research is needed.
    Gathers comprehensive research notes on the given subject.
    Output is passed to the Writer agent as reference material.
    """
    return Agent(
        role="Senior Research Analyst",                     # Job title - shapes the agent's identity
        goal="Research the given subject thoroughly and provide comprehensive, accurate information",
        backstory=(                                         # Backstory guides HOW the agent behaves
            "You are a meticulous researcher with expertise across many academic fields. "
            "You gather key concepts, facts, data, and relevant examples on any topic. "
            "You organize your research into clear, structured notes that a writer can use."
        ),
        llm=llm,                    # CrewAI LLM instance (uses litellm -> openrouter)
        allow_delegation=False,     # False = this agent does its own work, won't pass to others
        verbose=True,               # True = prints the agent's thinking process to console
        max_iter=5,                 # Max reasoning loops before stopping (prevents infinite loops)
    )


def create_writer(llm: LLM) -> Agent:
    """
    Writer Agent - called by manager after research is done.
    Takes the research and writes a structured academic assignment.
    If quality check fails, this agent rewrites with the feedback.
    """
    return Agent(
        role="Academic Assignment Writer",
        goal="Write a detailed, well-structured assignment based on the research provided",
        backstory=(
            "You are an experienced academic writer who creates clear, engaging, and "
            "well-organized assignments. You use proper structure with introduction, body sections, "
            "and conclusion. You cite facts from the research and explain concepts clearly."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
        max_iter=5,
    )


def create_quality_checker(llm: LLM) -> Agent:
    """
    Quality Checker Agent - called by manager after writing is done.
    Reviews the written assignment and gives a score from 1-10.
    If score < 7, the manager sends it back to Writer for revision.
    If score >= 7, the manager accepts the assignment as final.
    """
    return Agent(
        role="Quality Assurance Reviewer",
        goal="Review assignments for quality, accuracy, completeness, and structure",
        backstory=(
            "You are a strict academic reviewer with high standards. You check for: "
            "completeness, accuracy, structure (intro/body/conclusion), clarity, and depth. "
            "You give a quality score from 1-10 and specific feedback."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
        max_iter=5,
    )


def create_all_agents(llm: LLM) -> dict:
    """
    Creates all 3 sub-agents and returns them as a dictionary.
    Keys: "researcher", "writer", "quality_checker"
    These are only used when the manager decides to create an assignment.
    """
    return {
        "researcher": create_researcher(llm),
        "writer": create_writer(llm),
        "quality_checker": create_quality_checker(llm),
    }
