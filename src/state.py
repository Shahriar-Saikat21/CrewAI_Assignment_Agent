"""
state.py - Shared State
========================
Defines the shared state used during the assignment creation pipeline.
This is like a shared notebook that all sub-agents read from and write to.

Only used when the manager triggers the assignment pipeline.
Normal chat does NOT use this state.

State gets filled as each phase completes:
  Research phase  -> fills: research
  Writing phase   -> fills: draft
  Quality phase   -> fills: quality_feedback, quality_score
"""

from pydantic import BaseModel  # Pydantic ensures all fields have correct types


class AssignmentState(BaseModel):
    subject: str = ""              # The topic extracted from user's message
    research: str = ""             # Research notes from the Researcher agent
    draft: str = ""                # Written assignment from the Writer agent
    quality_feedback: str = ""     # Review feedback from Quality Checker
    quality_score: float = 0.0     # Score from 1-10 given by Quality Checker
    revision_count: int = 0        # How many times the Writer has revised (starts at 0)
    max_revisions: int = 2         # Max allowed revisions before accepting the draft as-is
