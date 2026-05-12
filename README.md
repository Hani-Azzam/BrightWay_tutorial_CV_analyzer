# CV Analyser & Hiring Decision Agent

An AI-powered hiring assistant built with Python, LangChain, Pinecone, and Google Gemini. The system has two phases: a conversational CV analyser for recruiters, and a multi-candidate hiring agent that scores, compares, and recommends candidates.

## Tech Stack

- **LLM** — Google Gemini (via LangChain)
- **Embeddings** — `models/gemini-embedding-001`
- **Vector Database** — Pinecone (serverless)
- **Framework** — LangChain, LangGraph
- **Language** — Python 3.13

---

## Architecture

```
AgentBase                        ← abstract contract (chat, reset, with-statement)
    │
    ├── RagAgent                 ← single-corpus RAG agent with memory & audit log
    │       └── CVAnalyserAgent  ← Phase 1: CV-specific commands on top of RagAgent
    │
    └── HiringAgent              ← Phase 2: composes multiple DocumentStores directly
```

**Key design decision:** `HiringAgent` extends `AgentBase` directly rather than `RagAgent` because it needs one isolated Pinecone namespace per candidate. `RagAgent` is designed around a single `DocumentStore` — extending it would mean constantly swapping the store in and out. Instead, `HiringAgent` composes the services directly and manages a `dict[str, DocumentStore]`.

---

## Project Structure

```
Assignment/
  agents/
    cv_analyser_agent.py   — CVAnalyserAgent: extends RagAgent, adds CV-specific commands
    hiring_agent.py        — HiringAgent: extends AgentBase, composes multiple stores
    rag_agent.py           — RagAgent: base single-corpus agent with memory & audit log
    conversation_agent.py  — tutorial agent (reference)
  base/
    agent_base.py          — abstract base class for all agents
    memory_base.py         — abstract base for memory implementations
    retriever_base.py      — abstract base for retrieval implementations
  services/
    document_store.py      — Pinecone-backed vector store with chunking
    embedding_service.py   — Gemini embedding wrapper
    llm_client.py          — Gemini LLM wrapper
    rag_pipeline.py        — retrieval + generation pipeline with refuse threshold
    vector_memory_store.py — in-memory vector store for conversation history
  data/
    cv_alice.txt           — fictional CV #1 (DataCard: pii_risk=high)
    cv_bob.txt             — fictional CV #2 (DataCard: pii_risk=high)
    job_description.txt    — shared JD with 5 required + 3 nice-to-have qualifications
  main_phase1.py           — Phase 1 REPL entry point
  main_phase2.py           — Phase 2 REPL entry point
  requirements.txt
  .env.example
```

---

## Phase 1 — CV Analyser Agent

Indexes a single CV into Pinecone and lets a recruiter explore it through natural language. Chat memory is active throughout the session.

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # fill in your API keys
python main_phase1.py
```

### Commands

| Command | Description |
|---|---|
| `summary` | Structured summary: name, experience level, top 3 skills |
| `skills` | Two lists: technical skills and soft skills |
| `improve <SECTION>` | Three specific improvement suggestions for a named section |
| `gap` | Top 3 skill/experience gaps vs the job description (Bonus A) |
| `memory <query>` | Search conversation memory directly |
| `exit` | Save audit log and quit |

Any other input is treated as a free-form question answered via RAG.

### Design Decisions

**Chunk size:** `chunk_size=200` with `\n--------` as the primary separator. CV sections are short and structured — larger chunks would slice across section boundaries and hurt retrieval accuracy.

**PII handling:** CV files are tagged `pii_risk=high` (name, email, phone, work history). User inputs are redacted with `_redact_pii()` before being stored in vector memory. The audit log records every turn for compliance.

---

## Phase 2 — Hiring Decision Agent

Indexes multiple CVs and a job description into isolated Pinecone namespaces, scores each candidate across multiple dimensions, and recommends who to invite for interview.

### Run

```bash
python main_phase2.py
```

### Commands

| Command | Description |
|---|---|
| `score` | Numeric scores for all candidates across 3+ dimensions |
| `compare` | Side-by-side table sorted by total score |
| `recommend` | Top candidate with written justification |
| `interview <name>` | Targeted interview questions for a candidate (Bonus B) |
| `exit` | Delete all candidate namespaces and quit |

### Design Decisions

**Namespace isolation:** Each candidate occupies a distinct Pinecone namespace (`candidate_alice`, `candidate_bob`). Retrieval for scoring queries only one namespace at a time — candidates never interfere with each other.

**Scoring dimensions:** Technical skills match, relevant experience, LLM/AI specialisation. Each dimension is scored 1–10 and the total determines ranking.

**Cleanup:** All candidate namespaces are deleted on exit via `store.clear()` — the Pinecone index itself is preserved.

---

## Responsible AI

| Practice | Implementation |
|---|---|
| DataCards | Every document tagged with source, license, PII risk, refresh cadence |
| PII redaction | Emails and phone numbers stripped from user input before storage |
| Refuse threshold | RAG pipeline refuses to answer when best chunk scores below 0.60 |
| Audit log | Every conversation turn saved to `data/audit_log.json` on exit |
| Contradiction detection | Agent flags when a new answer conflicts with memory |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
GEMINI_API_KEY=
GEMINI_MODEL_NAME=gemini-2.5-flash
GEMINI_TEMPERATURE=0.7
GEMINI_EMBEDDING_MODEL_NAME=models/gemini-embedding-001
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
```
