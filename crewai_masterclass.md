# CrewAI Masterclass — Complete Guide

> From zero to professional. Every concept, pattern, and trick you need to build production-grade multi-agent AI systems.

---

## Table of Contents

1. [What is CrewAI?](#1-what-is-crewai)
2. [Core Concepts](#2-core-concepts)
3. [Agents](#3-agents-in-depth)
4. [Tasks](#4-tasks-in-depth)
5. [Tools](#5-tools)
6. [Crews & Process Types](#6-crews--process-types)
7. [Memory & Context](#7-memory--context)
8. [CrewAI Flows (Advanced)](#8-crewai-flows-advanced)
9. [LLM Configuration](#9-llm-configuration)
10. [Patterns & Pro Tips](#10-patterns--pro-tips)
11. [Cheatsheet](#11-cheatsheet)

---

## 1. What is CrewAI?

CrewAI is a Python framework for building **multi-agent AI systems** where specialized agents collaborate to complete complex tasks.

Think of it like hiring a team. Instead of one AI trying to do everything, you create **specialists** — a researcher, a writer, a coder, a reviewer — each with their own role, tools, and focus. They work together in a defined workflow.

```
User Input → Crew → Agent 1 + Agent 2 + Agent 3 → Final Output
```

### Why use multi-agent systems?

| Benefit | Explanation |
|---|---|
| **Specialization** | Each agent is laser-focused on one job. Better results through focus. |
| **Parallelism** | Independent tasks can run at the same time. A crew can do in minutes what would take one agent hours. |
| **Quality control** | Agents can review each other's work. A critic agent catching errors before final output is a game-changer. |
| **Modularity** | Add, remove, or swap agents without rewriting everything. |

### CrewAI vs alternatives

| Framework | Best for |
|---|---|
| **CrewAI** | Role-based agents, task pipelines, production-ready structured workflows |
| **LangGraph** | Graph-based state machines. More control, more code. Complex branching logic. |
| **AutoGen** | Conversation-based multi-agent. Agents talk to each other. Good for research/prototyping. |

> **Recommendation:** CrewAI is the best starting point for most projects. It strikes the right balance between power and simplicity.

### Install

```bash
# Install CrewAI with all tools
pip install crewai crewai-tools

# Or just the core
pip install crewai
```

---

## 2. Core Concepts

The **5 building blocks** of every CrewAI project.

### 1. Agent — The worker

An AI entity with a role, goal, and backstory. Each agent is an expert in something. It uses an LLM as its brain and can be given tools to take actions (search the web, run code, read files, etc.).

### 2. Task — The unit of work

A specific job assigned to an agent. Has a `description` (what to do), `expected_output` (what to produce), and is assigned to an agent. Tasks can receive context from previous tasks.

### 3. Tool — Superpowers

Functions agents can call to interact with the world — search Google, read files, run Python, query databases, call APIs. Tools extend what an agent can do beyond just generating text.

### 4. Crew — The team

A collection of agents and tasks with a defined process. The crew orchestrates how agents work together — sequentially (one after another) or hierarchically (manager delegates to workers).

### 5. Process — The workflow

How tasks are executed. `sequential` = tasks run in order. `hierarchical` = a manager agent decides what to delegate. You can also use CrewAI Flows for complex conditional logic.

### How they connect

```python
from crewai import Agent, Task, Crew, Process

# 1. Create agents (workers)
researcher = Agent(role="Researcher", goal="Find facts", ...)
writer     = Agent(role="Writer",     goal="Write clearly", ...)

# 2. Define tasks (work units)
task1 = Task(description="Research topic X", agent=researcher, ...)
task2 = Task(description="Write about X",    agent=writer, context=[task1], ...)

# 3. Assemble the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential
)

# 4. Run it
result = crew.kickoff()
```

---

## 3. Agents in Depth

### Parameters

```python
from crewai import Agent

agent = Agent(
    # REQUIRED — the 3 most important fields
    role="Senior Data Analyst",        # Job title — shapes identity
    goal="Extract actionable insights", # What success looks like
    backstory="You have 10 years of...", # Context that guides behavior

    # LLM
    llm=my_llm,

    # TOOLS — what the agent can DO
    tools=[search_tool, file_read_tool],

    # BEHAVIOR CONTROLS
    verbose=True,           # Print agent's thinking
    allow_delegation=False, # Can it ask other agents for help?
    max_iter=15,            # Max reasoning loops (default 15)
    max_rpm=10,             # API calls per minute (rate limiting)

    # MEMORY
    memory=True,            # Remember things across tasks

    # CALLBACKS
    step_callback=my_fn,    # Called on each reasoning step
)
```

### Design Principles

**1. One job, done well**

Each agent should have exactly ONE primary responsibility. Avoid "do everything" agents — they produce mediocre results. A researcher researches. A writer writes. A reviewer reviews.

**2. Backstory shapes behavior**

The backstory is not decoration. `"You are a strict academic with 20 years experience reviewing PhD papers"` will review VERY differently than `"You are a friendly writing assistant"`. Use it deliberately.

**3. Tools match the role**

Only give an agent the tools it actually needs. Giving a writer access to a code execution tool is confusing and potentially dangerous. Match tools to responsibilities.

**4. `allow_delegation` with care**

When `True`, an agent can ask other agents for help. Useful for manager agents in hierarchical setups. For most worker agents, set it to `False` to avoid unexpected delegation chains.

> **Golden rule:** The best agent designs come from asking: *"If this were a real human job description, what would it say?"*

### Well-designed agent examples

```python
# Research agent
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive, accurate, up-to-date information",
    backstory="""You are a meticulous researcher trained at a top university.
    You verify sources, cross-reference facts, and only report
    information you are confident in. You always cite sources.""",
    tools=[SerperDevTool(), WebsiteSearchTool()],
    allow_delegation=False,
)

# Coding agent
coder = Agent(
    role="Senior Python Developer",
    goal="Write clean, tested, well-documented Python code",
    backstory="""Expert Python dev with 8 years experience.
    You follow PEP8, write type hints, always add docstrings,
    and think about edge cases before writing code.""",
    tools=[CodeInterpreterTool()],
)
```

### Advanced agent patterns

**Custom LLM per agent — different models for different tasks:**

```python
simple_agent = Agent(..., llm=get_llm("meta-llama/llama-3.1-8b"))
smart_agent  = Agent(..., llm=get_llm("openai/gpt-4o"))
```

**Dynamic agent creation:**

```python
def make_expert(domain: str) -> Agent:
    return Agent(
        role=f"{domain} Expert",
        goal=f"Provide deep expertise in {domain}",
        backstory=f"You have 15 years in {domain}...",
        llm=llm,
    )
```

**Step callback for monitoring:**

```python
def log_step(step_output):
    print(f"Step: {step_output.thought}")
    # Log to DB, send to monitoring, etc.

agent = Agent(..., step_callback=log_step)
```

---

## 4. Tasks in Depth

### Parameters

```python
from crewai import Task

task = Task(
    # REQUIRED
    description="""
    Research the latest trends in quantum computing.
    Focus on: applications in cryptography, timeline to
    practical use, and top companies leading development.
    Output structured notes with source citations.
    """,
    expected_output="""
    Structured research notes covering:
    - 3 key applications in cryptography
    - Realistic timeline (with expert citations)
    - Top 5 companies with brief profiles
    - Reference list (min 5 sources)
    """,
    agent=researcher,

    # CONTEXT — feed outputs from other tasks
    context=[previous_task],

    # OUTPUT — save result to a file
    output_file="research_notes.md",

    # STRUCTURED OUTPUT — get a Pydantic model back
    output_pydantic=ResearchReport,

    # CALLBACK — called when task completes
    callback=on_task_done,

    # ASYNC — run in parallel with other async tasks
    async_execution=False,
)
```

### Writing Good Tasks

**Bad task description:**

```python
description="Research quantum computing and write about it"
expected_output="A good report"
```

This is too vague. The agent doesn't know what to focus on, how long to be, what format to use, or what "good" means.

**Good task description:**

```python
description="""
Research quantum computing for a business executive audience.
Focus ONLY on: commercial applications ready within 5 years,
investment landscape (top 10 companies, funding rounds),
and regulatory risks.
Ignore: technical physics details, academic research.
Length: 600-800 words.
"""
expected_output="""
Markdown document with:
- Executive summary (100 words)
- 3 sections with subheadings
- Investment table (company, funding, focus area)
- Risk summary (3 bullet points)
"""
```

> **Rule:** Treat task descriptions like prompts. The more specific you are about what you want AND what you don't want, the better the output.

> **Warning:** Avoid using "you" in task descriptions — it can confuse the agent. Prefer imperative: "Research X", "Write Y", "Analyze Z".

### Passing Context Between Tasks

Context lets a downstream task see the output of upstream tasks. This is how you chain agents together.

```python
# Task 1: Research
research_task = Task(
    description="Research climate change impacts on agriculture",
    expected_output="Detailed research notes with data",
    agent=researcher,
)

# Task 2: Write (uses research_task output as context)
write_task = Task(
    description="Write a 1000-word article using the research provided",
    expected_output="Polished article in markdown",
    agent=writer,
    context=[research_task],  # <-- writer sees researcher's output
)

# Task 3: Review (uses both previous tasks)
review_task = Task(
    description="Review the article for accuracy against the research",
    expected_output="Reviewed and corrected article",
    agent=critic,
    context=[research_task, write_task],  # <-- sees both
)
```

> **Note:** In sequential process, each task automatically gets the previous task's output in its context. The explicit `context=[]` is most useful in hierarchical process or when you want non-adjacent tasks to share context.

### Structured Output with Pydantic

Instead of raw text, get back a typed Python object.

```python
from pydantic import BaseModel
from typing import List

class ResearchReport(BaseModel):
    title: str
    summary: str
    key_findings: List[str]
    sources: List[str]
    confidence_score: float

task = Task(
    description="Research and structure findings on solar energy",
    expected_output="Structured research report",
    agent=researcher,
    output_pydantic=ResearchReport,  # <-- magic here
)

result = crew.kickoff()

# Access structured fields directly!
print(result.pydantic.title)
print(result.pydantic.key_findings)
print(result.pydantic.confidence_score)
```

### Save Output to File

```python
task = Task(
    ...,
    output_file="final_report.md"  # auto-saved when task completes
)
```

---

## 5. Tools

Tools give agents the ability to act — search the web, read files, run code, call APIs.

### Built-in Tools

```python
from crewai_tools import (
    SerperDevTool,          # Google search (needs SERPER_API_KEY)
    WebsiteSearchTool,      # Semantic search within a website
    ScrapeWebsiteTool,      # Scrape full web page content
    FileReadTool,           # Read files from disk
    DirectoryReadTool,      # List/read directory contents
    CodeInterpreterTool,    # Execute Python code safely
    CSVSearchTool,          # Semantic search in CSV files
    PDFSearchTool,          # Semantic search in PDFs
    YoutubeVideoSearchTool, # Search YouTube transcripts
    GithubSearchTool,       # Search GitHub repositories
)

# Usage
search = SerperDevTool()
agent = Agent(tools=[search], ...)
```

### Creating Custom Tools

**Method 1: `@tool` decorator (simple)**

```python
from crewai.tools import tool

@tool("Weather Fetcher")
def get_weather(city: str) -> str:
    """Fetch current weather for a given city.
    Args: city (str): The city name to get weather for.
    Returns: Weather description string.
    """
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()["description"]

agent = Agent(tools=[get_weather], ...)
```

**Method 2: `BaseTool` class (full control)**

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City to get weather for")

class WeatherTool(BaseTool):
    name: str = "Weather Fetcher"
    description: str = "Gets current weather for any city"
    args_schema: type = WeatherInput

    def _run(self, city: str) -> str:
        # Your tool logic here
        return f"Sunny in {city}, 25°C"

agent = Agent(tools=[WeatherTool()], ...)
```

> **Use `BaseTool`** when you need input validation, caching, error handling, or want to share the tool across many agents cleanly.

### Tool Best Practices

**1. Write descriptive docstrings**

The agent reads the docstring to decide when and how to use the tool. A bad docstring means the agent won't use it or will use it wrong. Be explicit about inputs, outputs, and when to call it.

**2. Handle errors gracefully**

Tools should never crash the crew. Wrap network calls in try/except and return meaningful error strings so the agent can recover or try again.

**3. Return structured text**

Tools return strings. Format them clearly — JSON, markdown tables, or plain sentences. The better formatted your tool output, the better the agent can reason about it.

```python
# Error-safe tool pattern
@tool("Database Query")
def query_db(sql: str) -> str:
    """Execute a read-only SQL query. Returns results as JSON string."""
    try:
        results = db.execute(sql)
        return json.dumps(results)
    except Exception as e:
        return f"Error executing query: {str(e)}. Try again with a simpler query."
```

---

## 6. Crews & Process Types

### Sequential Process

Tasks run in order. Each task waits for the previous one to finish. Simple, predictable, most common.

```
Task 1 (Researcher) → Task 2 (Writer) → Task 3 (Critic) → Result
```

```python
crew = Crew(
    agents=[researcher, writer, critic],
    tasks=[task1, task2, task3],
    process=Process.sequential,  # default
    verbose=True,
)
result = crew.kickoff()
```

> **Note:** In sequential mode, every task automatically receives the output of the PREVIOUS task as context. You don't need to set `context=` manually unless you want non-adjacent task context.

### Hierarchical Process

A manager agent reads the overall goal and dynamically delegates subtasks to worker agents. More flexible but more expensive (extra LLM calls for the manager).

```
          Manager Agent
         /      |      \
   Worker 1  Worker 2  Worker 3
```

```python
manager = Agent(
    role="Project Manager",
    goal="Coordinate the team to produce the best report",
    backstory="Expert at breaking down tasks and delegating",
    allow_delegation=True,  # REQUIRED for manager
    llm=llm,
)

crew = Crew(
    agents=[researcher, writer, critic],
    tasks=[main_task],          # Manager breaks it down
    process=Process.hierarchical,
    manager_agent=manager,      # OR use manager_llm=llm
    verbose=True,
)
```

> **Warning:** Hierarchical mode uses significantly more tokens. Use it when tasks are genuinely unpredictable or interdependent. For simple pipelines, sequential is cheaper and faster.

### Full Crew Configuration

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential,

    # VERBOSITY
    verbose=True,          # Print agent thoughts (2 = full detail)

    # MEMORY
    memory=True,           # Enable crew-level memory

    # CACHING
    cache=True,            # Cache tool results (saves $$$)

    # RATE LIMITING
    max_rpm=10,            # Max LLM requests per minute

    # LANGUAGE
    language="English",    # Force output language

    # CALLBACKS
    step_callback=fn,      # Called each agent step
    task_callback=fn,      # Called each task completion
)

# Kickoff with dynamic inputs
result = crew.kickoff(inputs={"topic": "quantum computing"})

# Kickoff with list of inputs (runs crew once per item)
results = crew.kickoff_for_each(inputs=[
    {"topic": "AI"},
    {"topic": "Blockchain"},
])
```

### Working with Crew Output

```python
result = crew.kickoff()

# Raw string output
print(result.raw)

# Token usage stats
print(result.token_usage)

# Individual task outputs
for task_output in result.tasks_output:
    print(task_output.description)
    print(task_output.raw)
    print(task_output.agent)

# If task used output_pydantic
print(result.pydantic)       # Pydantic model instance
print(result.json_dict)      # Dict version
```

---

## 7. Memory & Context

How CrewAI agents remember things — within a task, across tasks, and across crew runs.

| Memory Type | Description |
|---|---|
| **Short-term** | Within a single crew run. Agents remember what happened in earlier tasks. |
| **Long-term** | Persists across crew runs (stored in SQLite by default). |
| **Entity memory** | Tracks entities (people, places, companies) mentioned across tasks. |
| **Contextual** | Combines short + long-term to give relevant context at each step. |

### Enabling Memory

```python
# Enable all memory types
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,           # turns on short + long + entity
    embedder={             # optional: configure the embedder
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    }
)

# Use OpenRouter for embeddings (alternative)
from langchain_openai import OpenAIEmbeddings
embedder = OpenAIEmbeddings(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    model="openai/text-embedding-3-small"
)
```

### User Memory (external knowledge)

```python
from crewai.memory import UserMemory

memory = UserMemory()
memory.save("user_name", "Saikat")
memory.save("project", "SleepAI mobile app")
memory.save("stack", "React Native, AWS Amplify")

crew = Crew(..., user_memory=memory)
```

### Passing Dynamic Inputs (most common pattern)

```python
# Use {variable} placeholders in descriptions
task = Task(
    description="Research {topic} focusing on {angle}. Audience: {audience}",
    expected_output="Report suitable for {audience}",
    agent=researcher,
)

# Pass values at runtime
result = crew.kickoff(inputs={
    "topic": "machine learning",
    "angle": "practical business applications",
    "audience": "non-technical executives",
})
```

> **Tip:** For most projects, simply set `memory=True` on the Crew. It dramatically improves consistency across long runs. The default SQLite storage works fine for local development.

---

## 8. CrewAI Flows (Advanced)

Flows let you add **conditional logic, loops, and state management** on top of Crews. Think of Flows as the conductor that manages multiple Crews — each Crew is a musician who plays when the conductor cues them.

Use Flows when a simple crew isn't enough — when you need "if research finds X, run crew A, otherwise run crew B", or when you need to loop until a condition is met.

### Full Flow Example

```python
from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel

class AssignmentState(BaseModel):
    topic: str = ""
    research_done: bool = False
    quality_score: float = 0.0
    draft: str = ""
    final: str = ""

class AssignmentFlow(Flow[AssignmentState]):

    @start()
    def research_phase(self):
        print(f"Researching: {self.state.topic}")
        result = research_crew.kickoff(
            inputs={"topic": self.state.topic}
        )
        self.state.draft = result.raw
        self.state.research_done = True

    @listen(research_phase)
    def score_quality(self):
        score = evaluate_quality(self.state.draft)
        self.state.quality_score = score

    @router(score_quality)
    def decide_next(self):
        if self.state.quality_score >= 7.0:
            return "finalize"    # go to finalize method
        else:
            return "improve"     # go to improve method

    @listen("improve")
    def improve(self):
        result = improvement_crew.kickoff(
            inputs={"draft": self.state.draft}
        )
        self.state.draft = result.raw
        self.state.quality_score = 0.0  # reset, re-score

    @listen("finalize")
    def finalize(self):
        self.state.final = self.state.draft
        print("Assignment complete!")

# Run the flow
flow = AssignmentFlow()
flow.kickoff(inputs={"topic": "quantum computing"})
```

### Key Flow Decorators

| Decorator | Purpose |
|---|---|
| `@start()` | Entry point(s) of the flow. Can have multiple start methods that run in parallel. |
| `@listen(method)` | Runs after the specified method completes. `@listen(a, b)` = runs after BOTH complete. |
| `@router(method)` | Conditional routing. Return a string matching a method name to control flow direction. |
| `@and_(a, b)` | Runs only after BOTH methods finish (AND gate). Great for parallel crews merging. |

---

## 9. LLM Configuration

### OpenRouter (recommended)

OpenRouter is an API router that gives access to 100+ models through one endpoint. It's OpenAI-API-compatible, so CrewAI works seamlessly.

```python
from langchain_openai import ChatOpenAI
import os

def get_llm(model: str = "openai/gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",  # required
            "X-Title": "MyApp",                  # optional
        },
        temperature=0.7,   # 0 = deterministic, 1 = creative
        max_tokens=4096,   # limit output length
    )

# Use different models for different agents
cheap_llm  = get_llm("meta-llama/llama-3.1-8b-instruct")  # fast + cheap
smart_llm  = get_llm("openai/gpt-4o")                     # powerful
claude_llm = get_llm("anthropic/claude-3.5-sonnet")        # great for writing
```

### Local Models

**Ollama (free, runs locally):**

```python
# 1. Install Ollama: https://ollama.ai
# 2. Pull a model: ollama pull llama3.1
# 3. Connect CrewAI:

from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3.1",
    base_url="http://localhost:11434",
)
agent = Agent(..., llm=llm)
```

**LM Studio (GUI, easy):**

```python
# Run LM Studio, load any GGUF model, start local server
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",  # any string works
    model="local-model",
)
```

### Model Selection Guide

| Use case | Recommended model |
|---|---|
| Best overall | `anthropic/claude-3.5-sonnet` — excellent instruction following, great for writer/editor agents |
| Best for reasoning | `openai/gpt-4o` — strongest logical reasoning and tool use, best for orchestrator/manager agents |
| Best free option | `meta-llama/llama-3.1-70b-instruct` — free tier on OpenRouter, surprisingly capable |
| Best for code | `openai/gpt-4o` or `deepseek/deepseek-coder` — DeepSeek Coder is free and very strong |

### Cost Optimization Tips

**1. Use cheaper models for simple agents**
A formatting agent doesn't need GPT-4o. Use `llama-3.1-8b` for agents doing simple, well-defined tasks.

**2. Enable crew caching**
`cache=True` on the Crew means if the same tool is called with the same input, the cached result is returned. Saves a lot on repeated runs during development.

**3. Limit `max_iter`**
Default is 15 iterations per agent. For simple tasks, set `max_iter=5` to prevent agents from spiraling into expensive thought loops.

**4. Use `max_rpm`**
Set `max_rpm=10` on the Crew or Agent to rate-limit API calls. Prevents accidental runaway spending.

```python
# Cost-optimized crew setup
researcher = Agent(..., llm=cheap_llm, max_iter=5)
crew = Crew(
    agents=[researcher, writer, critic],
    tasks=[...],
    cache=True,   # cache tool results
    max_rpm=10,   # rate limit
)
```

---

## 10. Patterns & Pro Tips

### Recommended Project Structure

```
my_crew/
├── main.py              # entry point, runs the crew
├── .env                 # API keys (never commit!)
├── requirements.txt
│
├── agents/
│   ├── __init__.py
│   ├── researcher.py    # each agent in its own file
│   ├── writer.py
│   └── critic.py
│
├── tasks/
│   ├── __init__.py
│   └── assignment_tasks.py
│
├── tools/
│   ├── __init__.py
│   └── custom_tools.py
│
└── crews/
    ├── __init__.py
    └── assignment_crew.py  # Crew assembly
```

```python
# crews/assignment_crew.py
from crewai import Crew, Process
from agents.researcher import researcher
from agents.writer import writer
from tasks.assignment_tasks import build_tasks

def create_crew(topic: str) -> Crew:
    tasks = build_tasks(topic)
    return Crew(
        agents=[researcher, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
```

### Debugging Techniques

**Step-by-step output:**

```python
crew = Crew(..., verbose=2)
```

**Inspect individual task outputs:**

```python
result = crew.kickoff()
for i, task_out in enumerate(result.tasks_output):
    print(f"\n--- Task {i+1}: {task_out.description[:50]} ---")
    print(task_out.raw[:500])  # first 500 chars
```

**Test agents in isolation:**

```python
# Test a single agent without the full crew
test_crew = Crew(
    agents=[researcher],
    tasks=[single_task],
    verbose=True,
)
result = test_crew.kickoff()
```

**Token usage tracking:**

```python
result = crew.kickoff()
usage = result.token_usage
print(f"Total tokens: {usage.total_tokens}")
print(f"Prompt: {usage.prompt_tokens}")
print(f"Completion: {usage.completion_tokens}")
```

### Production Checklist

**1. Always use `.env` for secrets**
Never hardcode API keys. Use `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()
```

**2. Add error handling around `kickoff()`**
Crew runs can fail due to API errors, rate limits, or bad outputs. Implement retry logic:

```python
import time

def safe_kickoff(crew, inputs, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = crew.kickoff(inputs=inputs)
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # exponential backoff
            print(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s")
            time.sleep(wait)
```

**3. Set timeouts**
Long-running crews can hang. Set `max_iter` on agents to prevent infinite loops.

**4. Log everything in production**
Use `step_callback` and `task_callback` to log to your observability platform (Langsmith, Langfuse, etc.).

### The Most Common CrewAI Mistakes

| Mistake | Fix |
|---|---|
| **Vague task descriptions** | Be specific about length, format, tone, what to include, and what to exclude. |
| **Too many agents** | 3–5 well-designed agents outperforms 10 vague ones. Start minimal. |
| **Ignoring `expected_output`** | The LLM reads this to know the format. Write it as specifically as the description. |
| **`allow_delegation=True` everywhere** | Keep it `False` on worker agents, `True` only on explicit manager agents. |
| **Not testing incrementally** | Build and test one agent/task at a time. Don't debug a 6-agent crew all at once. |

---

## 11. Cheatsheet

### Agent Parameters

| Parameter | Description |
|---|---|
| `role` | Job title — shapes the agent's identity and behavior |
| `goal` | What success looks like for this agent |
| `backstory` | Persona context — most underused but very powerful |
| `llm` | The LLM to use |
| `tools` | List of tools the agent can use |
| `verbose` | `True` = print agent's internal reasoning |
| `allow_delegation` | `True` = can delegate to other agents. Keep `False` for workers. |
| `max_iter` | Max reasoning loops before forced output (default: 15) |
| `memory` | `True` = enable memory for this agent |
| `step_callback` | Function called on each reasoning step |

### Task Parameters

| Parameter | Description |
|---|---|
| `description` | What to do — be very specific and detailed |
| `expected_output` | What to produce — format, length, structure. The LLM reads this! |
| `agent` | Which agent handles this task |
| `context` | List of tasks whose outputs this task can read |
| `output_file` | Auto-save output to this file path |
| `output_pydantic` | Pydantic model class — returns typed output |
| `async_execution` | `True` = run in parallel with other async tasks |
| `callback` | Function called when task finishes |

### Crew Parameters

| Parameter | Description |
|---|---|
| `agents` | List of Agent instances |
| `tasks` | List of Task instances (order matters for sequential) |
| `process` | `Process.sequential` or `Process.hierarchical` |
| `manager_agent` | Custom manager agent for hierarchical process |
| `memory` | `True` = enable short + long + entity memory |
| `cache` | `True` = cache tool results (saves money) |
| `max_rpm` | Rate limit: API calls per minute |
| `verbose` | `True` / `2` — print execution details |

### Built-in Tools

| Tool | Purpose |
|---|---|
| `SerperDevTool` | Google search. Needs `SERPER_API_KEY` |
| `ScrapeWebsiteTool` | Scrape full web page content |
| `FileReadTool` | Read local files |
| `CodeInterpreterTool` | Execute Python code in sandbox |
| `CSVSearchTool` | Semantic search in CSV files |
| `PDFSearchTool` | Semantic search in PDFs |
| `WebsiteSearchTool` | Semantic search within a website |
| `GithubSearchTool` | Search GitHub repositories |

### Key Commands

```python
crew.kickoff()                          # run crew once
crew.kickoff(inputs={"k": "v"})         # with dynamic inputs
crew.kickoff_for_each([{...}, {...}])    # run for each input set
crew.kickoff_async()                    # non-blocking async run

result.raw                              # final output as string
result.tasks_output                     # list of all task results
result.token_usage                      # token stats
result.pydantic                         # if output_pydantic was used
```

### OpenRouter Setup (quick copy)

```python
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="openai/gpt-4o",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "MyApp"},
    temperature=0.7,
)
```

---

## Suggested Learning Path

| Week | Focus |
|---|---|
| **Week 1** | Build 3 simple 2-agent crews on different topics |
| **Week 2** | Add custom tools, try Pydantic outputs, experiment with backstories |
| **Week 3** | Build a Flow with conditional routing (`@router` decorator) |
| **Week 4** | Deploy something real — wrap a crew in FastAPI or a CLI tool |

---

*CrewAI documentation: [docs.crewai.com](https://docs.crewai.com)*
*OpenRouter models: [openrouter.ai/models](https://openrouter.ai/models)*
