"""
tasks.py - Task Definitions
============================
Defines the work (tasks) that each sub-agent performs.
Each function creates a Task with a description (what to do) and expected_output (what to produce).

Tasks (only created when manager triggers the assignment pipeline):
  1. Research Task  - Tell researcher what to research
  2. Writing Task   - Tell writer what to write (includes feedback if revising)
  3. Quality Task   - Tell quality checker what to review and how to score
"""

from crewai import Agent, Task


def create_research_task(subject: str, researcher: Agent) -> Task:
    """
    Creates the research task.
    The researcher will gather all information needed to write the assignment.

    - subject: the topic to research (extracted from user's message by manager)
    - researcher: the agent that will execute this task
    """
    return Task(
        description=(                    # What the agent should do - be specific!
            f"Research the subject: '{subject}'\n\n"
            "Provide comprehensive research notes covering:\n"
            "- Key concepts and definitions\n"
            "- Important facts and data\n"
            "- Real-world examples or applications\n"
            "- Different perspectives or subtopics\n"
            "- Any relevant recent developments\n\n"
            "Make the research detailed enough to write a full assignment."
        ),
        expected_output="Detailed, structured research notes with key concepts, facts, examples, and sources",
        agent=researcher,                # Which agent does this task
    )


def create_writing_task(subject: str, research: str, writer: Agent, feedback: str = "") -> Task:
    """
    Creates the writing task.
    The writer uses the research to create the assignment.
    If this is a revision (feedback != ""), the writer also gets the quality feedback to fix.

    - subject: the topic
    - research: output from the researcher
    - writer: the agent that will execute this task
    - feedback: quality checker's feedback (empty on first write, filled on revisions)
    """
    # Only include feedback section if there's feedback (i.e., this is a revision)
    feedback_section = ""
    if feedback:
        feedback_section = (
            f"\n\nIMPORTANT - Previous review feedback to address:\n{feedback}\n"
            "Fix ALL issues mentioned in the feedback while keeping what was good."
        )

    return Task(
        description=(
            f"Write a detailed academic assignment on: '{subject}'\n\n"
            f"Use this research:\n{research}\n"           # Research from researcher
            f"{feedback_section}\n\n"                     # Feedback from Quality Checker (if revising)
            "Requirements:\n"
            "- Title\n"
            "- Introduction (set context, state purpose)\n"
            "- 3-5 body sections with subheadings\n"
            "- Each section should have explanations, examples, and analysis\n"
            "- Conclusion (summarize key points, final thoughts)\n"
            "- References section\n"
            "- Total length: 1000-1500 words"
        ),
        expected_output="A complete, well-structured academic assignment in markdown format",
        agent=writer,
    )


def create_quality_task(subject: str, draft: str, quality_checker: Agent) -> Task:
    """
    Creates the quality check task.
    The quality checker reviews the draft and scores it 1-10.
    Score >= 7 = approved, Score < 7 = needs revision.

    - subject: the topic (for context)
    - draft: the written assignment from the writer
    - quality_checker: the agent that will execute this task
    """
    return Task(
        description=(
            f"Review this assignment on '{subject}':\n\n"
            f"{draft}\n\n"                                # The assignment to review
            "Evaluate on these criteria (score each 1-10):\n"
            "1. Completeness - Are all aspects of the topic covered?\n"
            "2. Accuracy - Are facts and explanations correct?\n"
            "3. Structure - Is it well-organized with proper sections?\n"
            "4. Clarity - Is it easy to understand?\n"
            "5. Depth - Is it detailed enough?\n\n"
            "IMPORTANT: Start your response with EXACTLY this format:\n"
            "OVERALL_SCORE: X/10\n\n"                    # This format is parsed by utils.py
            "Then provide detailed feedback.\n"
            "If score is 7 or above, say 'APPROVED'.\n"
            "If below 7, list specific improvements needed."
        ),
        expected_output="Quality score (OVERALL_SCORE: X/10) followed by detailed feedback",
        agent=quality_checker,
    )
