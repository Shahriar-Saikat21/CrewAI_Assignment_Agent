# Assignment Agent - CrewAI Multi-Agent System

An AI-powered assignment creation tool built with [CrewAI](https://docs.crewai.com). It uses a **smart manager** that chats with the user and intelligently delegates work to **3 specialized sub-agents** only when needed.

---

## How It Works

```
User types a message
        |
        v
   ┌─────────┐
   │ Manager  │  ← Smart router: classifies every message
   └────┬─────┘
        │
        ├── "hello"              → Chat response (no agents used)
        ├── "what's the weather?" → Off-topic rejection (no agents used)
        └── "write assignment on AI" → Triggers assignment pipeline ↓

   ┌──────────────── Assignment Pipeline ─────────────────┐
   │                                                       │
   │  [Researcher]  → Gathers info on the topic            │
   │       |                                               │
   │  [Writer]      → Writes the full assignment           │
   │       |                                               │
   │  [Quality Checker] → Scores it 1-10                   │
   │       |                                               │
   │   Score >= 7? ──YES──→ Save to file → Done!           │
   │       |                                               │
   │      NO                                               │
   │       |                                               │
   │   Revisions < 2? ──YES──→ Send feedback to Writer ──┐ │
   │       |                          (loop back)         │ │
   │      NO                                              │ │
   │       |                                              │ │
   │   Accept current draft → Save to file → Done!        │ │
   │                                                       │
   └───────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── main.py              → Entry point: chat loop (input/output)
├── src/
│   ├── __init__.py
│   ├── config.py        → LLM setup (OpenRouter connection, free model config)
│   ├── manager.py       → Smart manager (intent detection, chat, pipeline orchestration)
│   ├── agents.py        → 3 sub-agent definitions (researcher, writer, quality checker)
│   ├── tasks.py         → Task templates for each agent
│   ├── state.py         → Shared state model (data passed between pipeline phases)
│   └── utils.py         → Helpers (score parsing, file saving)
├── requirements.txt     → Python dependencies
├── .env                 → Your OpenRouter API key (not committed to git)
├── .env.example         → Template for .env
└── .gitignore
```

---

## File Responsibilities

| File | What It Does |
|---|---|
| `main.py` | Takes user input in a loop, sends to manager, prints response |
| `src/config.py` | Creates 2 LLM instances: `ChatOpenAI` for chat, `CrewAI LLM` for agents |
| `src/manager.py` | The brain — classifies intent, handles chat, runs the assignment pipeline |
| `src/agents.py` | Defines 3 sub-agents with roles, goals, and backstories |
| `src/tasks.py` | Creates task descriptions for each agent (what to do, what to produce) |
| `src/state.py` | Pydantic model that holds shared data between pipeline phases |
| `src/utils.py` | Parses quality scores from text, saves assignments to `.md` files |

---

## The 3 Sub-Agents

| Agent | Role | When Called |
|---|---|---|
| **Researcher** | Gathers key concepts, facts, examples, and data on the topic | First — always runs when pipeline starts |
| **Writer** | Writes a structured assignment (intro, body, conclusion) using the research | After researcher — may run multiple times if quality is low |
| **Quality Checker** | Reviews the draft and scores it 1-10 on completeness, accuracy, structure, clarity, depth | After writer — decides if revision is needed |

---

## Intent Classification

The manager classifies every user message into one of 3 intents:

| Intent | Example Messages | What Happens |
|---|---|---|
| `chat` | "hello", "what can you do?", "thanks" | Manager responds using LLM directly (no agents) |
| `assignment` | "create assignment on AI", "write about climate change" | Manager extracts topic, triggers full pipeline |
| `off_topic` | "what's the weather?", "help me with Python code" | Static rejection message (no LLM call needed) |

---

## Two LLM Instances

The project uses **two separate LLM connections** to the same model:

| Instance | Type | Used By | Why |
|---|---|---|---|
| `chat_llm` | `ChatOpenAI` (LangChain) | Manager's direct chat | LangChain sends requests to OpenRouter URL directly — just needs URL + key |
| `crew_llm` | `LLM` (CrewAI) | Sub-agents (researcher, writer, quality checker) | CrewAI uses `litellm` internally, which needs `openrouter/` prefix to identify the provider |

```python
# Chat LLM — model name as-is
ChatOpenAI(model="nvidia/nemotron-3-super-120b-a12b:free", openai_api_base="https://openrouter.ai/api/v1")

# Crew LLM — needs "openrouter/" prefix for litellm
LLM(model="openrouter/nvidia/nemotron-3-super-120b-a12b:free")
```

---

## Quality Loop

The quality check creates a feedback loop:

1. **Writer** writes the assignment
2. **Quality Checker** scores it (1-10) and gives feedback
3. If score **>= 7** → approved, move to final output
4. If score **< 7** → feedback is sent back to the Writer, who rewrites
5. Maximum **2 revisions** — after that, the current draft is accepted regardless of score

The score is extracted from the Quality Checker's text output using regex:
```
"OVERALL_SCORE: 8/10" → parsed as 8.0
```

---

## Shared State

The `AssignmentState` is a Pydantic model that acts as a shared notebook between phases:

```
Phase 1 (Research)  → fills: state.research
Phase 2 (Write)     → fills: state.draft
Phase 3 (Quality)   → fills: state.quality_feedback, state.quality_score
```

| Field | Type | Description |
|---|---|---|
| `subject` | str | The topic from user input |
| `research` | str | Research notes from Researcher |
| `draft` | str | Written assignment from Writer |
| `quality_feedback` | str | Review feedback from Quality Checker |
| `quality_score` | float | Score 1-10 from Quality Checker |
| `revision_count` | int | How many times Writer has revised |
| `max_revisions` | int | Max revisions allowed (default: 2) |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- An [OpenRouter](https://openrouter.ai) account (free, no deposit needed)

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd CrewAI_Assignment_Agent
```

### Step 2: Create Virtual Environment

A virtual environment keeps this project's packages isolated from your system Python.

**Windows:**
```bash
python -m venv venv
```

**Mac/Linux:**
```bash
python3 -m venv venv
```

This creates a `venv/` folder inside the project.

### Step 3: Activate the Virtual Environment

You must activate it every time you open a new terminal to work on this project.

**Windows (CMD):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Windows (Git Bash):**
```bash
source venv/Scripts/activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

When activated, you'll see `(venv)` at the start of your terminal prompt:
```
(venv) E:\...\CrewAI_Assignment_Agent>
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:
- `crewai` — multi-agent framework
- `crewai-tools` — agent tools
- `langchain-openai` — LLM chat client
- `litellm` — provider routing for CrewAI agents
- `python-dotenv` — loads `.env` file

### Step 5: Get Your OpenRouter API Key

1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up (Google/GitHub login works) — **no deposit or credit card needed**
3. Go to **Keys** → **Create Key**
4. Copy the key

### Step 6: Add API Key to `.env`

Open the `.env` file in the project root and paste your key:

```
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

### Step 7: Run the Project

```bash
python main.py
```

You should see:
```
============================================================
  Assignment Agent
  Chat normally or ask me to create an assignment!
  Type 'quit' to exit.
============================================================

You:
```

Type a message and press Enter. Type `quit` to exit.

### Deactivate Virtual Environment (when done)

```bash
deactivate
```

---

## Usage Examples

```
============================================================
  Assignment Agent
  Chat normally or ask me to create an assignment!
  Type 'quit' to exit.
============================================================

You: hello
  [Manager] Detected intent: chat
Assistant: Hi! I create academic assignments. Give me a topic!

You: what's the weather today?
  [Manager] Detected intent: off_topic
Assistant: Sorry, I can only help with creating academic assignments.
           Tell me a subject and I'll create one for you!

You: create an assignment on artificial intelligence
  [Manager] Detected intent: assignment
  [Manager] Extracted subject: 'Artificial Intelligence'
  [Manager] Starting assignment pipeline...
  [Researcher] Researching 'Artificial Intelligence'...
  [Writer] Writing assignment (Revision #0)...
  [Quality Checker] Reviewing assignment...
  >> Quality Score: 8.0/10
  >> APPROVED!
  DONE! Assignment saved to: assignment_artificial_intelligence.md

You: quit
Goodbye!
```

---

## Changing the Free Model

Edit `FREE_MODEL` in `src/config.py`:

```python
FREE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
```

Other free alternatives on OpenRouter:
- `meta-llama/llama-3.1-8b-instruct:free`
- `google/gemma-2-9b-it:free`

Check [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free) for the latest free models.

---

## Dependencies

| Package | Purpose |
|---|---|
| `crewai` | Multi-agent framework (agents, tasks, crews) |
| `crewai-tools` | Built-in tools for agents (optional, for future use) |
| `langchain-openai` | ChatOpenAI client for manager's direct chat |
| `litellm` | Lets CrewAI talk to OpenRouter (provider routing) |
| `python-dotenv` | Loads `.env` file into environment variables |

---

## Tech Stack

- **Framework**: [CrewAI](https://docs.crewai.com) — multi-agent orchestration
- **LLM Provider**: [OpenRouter](https://openrouter.ai) — API gateway for free AI models
- **LLM Routing**: [LiteLLM](https://docs.litellm.ai) — maps model names to providers
- **Chat Client**: [LangChain OpenAI](https://python.langchain.com) — direct LLM calls for chat
- **Language**: Python 3.10+
