# NovaBite Multi-Agent RAG System

Production-style multi-agent assistant for **NovaBite Restaurants** using LangChain, RAG, sub-agents, tool execution, memory, and orchestration.

## Requirement Coverage

- **Main orchestrator agent**: `src/agents.py::MainOrchestratorAgent`
  - intent classification (`rag`, `operations`, `mixed`, `clarify`)
  - routes to sub-agents only (no direct business tool execution)
  - conversation memory across turns
  - response merge for mixed requests
  - validation / safe fallback behavior
  - clarification when inputs are ambiguous or incomplete
- **Sub-agent A (RAG)**: `src/agents.py::RestaurantKnowledgeRAGAgent`
  - answers from internal docs only
  - implemented domains: menu/allergens, opening hours, event hosting
  - menu recommendation support (`recommend`/`suggest` intents)
- **Sub-agent B (Operations)**: `src/agents.py::OperationsAgent`
  - tool-based operational flows with simulated backend behavior
- **Tools implemented (>=2 required, 4 implemented)**:
  - `check_table_availability(date, time, branch)`
  - `book_table(name, date, time, branch)`
  - `get_today_special(branch)`
  - `check_loyalty_points(user_id)`
- **RAG evaluation**: `src/evaluate_retrieval.py`
  - retrieval hit checks for core benchmark queries

## Architecture

- `src/main.py`: CLI entrypoint
- `src/agents.py`: orchestrator + two sub-agents + memory + guardrails
- `src/rag_pipeline.py`: ingestion, chunking, embeddings, FAISS retrieval
- `src/tools.py`: operational tool implementations (simulated MCP-style server logic)
- `src/build_index.py`: build/rebuild vector index
- `data/knowledge/*.md`: internal knowledge base corpus

## RAG Design Decisions

- **Ingestion pipeline**
  - source format: markdown docs in `data/knowledge/`
  - loader: `TextLoader`
- **Chunking strategy**
  - `RecursiveCharacterTextSplitter`
  - `chunk_size=500`, `chunk_overlap=80`
  - section-aware separators (`##`, `###`, bullets) to preserve semantic blocks
- **Embedding model**
  - `sentence-transformers/all-MiniLM-L6-v2`
  - chosen for fast local inference and strong semantic retrieval for short FAQ-like queries
- **Vector database**
  - FAISS (`langchain_community.vectorstores.FAISS`)
  - local, reproducible, and fast for evaluation
- **Retrieval strategy**
  - top-k retrieval (`k=4`) with lightweight term-aware ordering
- **Context filtering**
  - section-level chunks with source metadata (`source`, `chunk_id`)
- **Hallucination prevention**
  - strict document-grounded answering
  - deterministic extractive response path for tiny OSS model stability
  - explicit fallback when evidence is missing
- **Grounded answer generation**
  - answers are assembled from retrieved evidence only; unknown items are not fabricated

## Memory Design

- Turn history is stored in orchestrator (`self.history`) as chat messages.
- Follow-up queries use prior turns for routing and continuity.
- Example continuity:
  - ask information first, then booking request in next turn
  - context is retained by the orchestrator across turns

## Tool Simulation / MCP-style Notes

- Tools in `src/tools.py` simulate external service calls with strict input schemas.
- Operations agent does tool routing, executes exactly one tool per operation query, then returns user-safe output.
- Booking state is persisted in-process via `BOOKINGS` for session realism.

## Setup and Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment:

```bash
copy .env.example .env
```

Default model is open-source local:

```bash
OSS_MODEL_NAME=sshleifer/tiny-gpt2
```

3. Build index:

```bash
python -m src.build_index
```

4. Run retrieval evaluation:

```bash
python -m src.evaluate_retrieval
```

5. Start assistant:

```bash
python -m src.main
```

## Example Queries and Outputs

- **RAG**
  - `Do you have vegan pasta?`
  - `Is the chicken grilled or fried?`
  - `What are your opening hours on weekends?`
  - `Do you host birthday events?`
  - `Can you recommend a vegan option?`
- **Operations**
  - `Check table availability for 2026-05-10 at 19:00 in Downtown.`
  - `Book a table for Sara on 2026-05-10 at 19:00 in Downtown.`
  - `What is today's special in Riverside?`
  - `How many loyalty points does NB-1001 have?`

## Assumptions

- Tool backend is simulated (MCP-style behavior, local implementation).
- Single-process in-memory state is acceptable for this evaluation scope.
- Knowledge docs are authoritative; unsupported items are treated as unverified.
- Tiny OSS model is used for local compatibility; deterministic routing/extraction protects reliability.
