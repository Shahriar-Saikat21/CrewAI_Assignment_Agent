"""
utils.py - Helper Functions
============================
Small utility functions used by the manager.

Functions:
  - parse_quality_score: Extracts the numeric score (X/10) from quality checker's text output
  - save_assignment: Saves the final assignment to a .md file
"""

import re


def parse_quality_score(feedback: str) -> float:
    """
    Extracts the quality score from the quality checker's feedback text.

    The quality checker is told to write "OVERALL_SCORE: X/10" in its response.
    This function finds that pattern and returns the number.

    Example: "OVERALL_SCORE: 8/10" -> returns 8.0
    Example: "Score: 6.5/10"       -> returns 6.5

    Returns 7.0 as default if parsing fails (to avoid infinite revision loops).
    """
    for line in feedback.split("\n"):
        if "SCORE" in line.upper():                              # Find the line containing "SCORE"
            numbers = re.findall(r"(\d+(?:\.\d+)?)\s*/\s*10", line)  # Extract "X/10" pattern
            if numbers:
                return float(numbers[0])
    return 7.0  # Default: assume pass if we can't parse the score


def save_assignment(subject: str, content: str) -> str:
    """
    Saves the final assignment content to a markdown file.

    - subject: used to generate the filename (e.g., "Machine Learning" -> "assignment_machine_learning.md")
    - content: the full assignment text to save

    Returns the filename that was created.
    """
    safe_name = subject.replace(" ", "_").lower()[:30]   # Make filename safe (no spaces, max 30 chars)
    filename = f"assignment_{safe_name}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename
