# 🚀 LangGraph Enhanced RAG - Self-Corrective Customer Support Assistant

> **Branch:** `rag-langgraph`  
> **Status:** ✅ Complete  
> **Upgrade from:** Basic RAG → Advanced Self-Corrective RAG

This branch enhances the basic RAG system with **LangGraph** to create a self-corrective, agentic workflow with document grading, hallucination detection, and automatic regeneration.

---

## 📊 Architecture Overview

### Graph Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LangGraph Self-Corrective RAG                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   START                                                                  │
│     │                                                                    │
│     ▼                                                                    │
│   ┌──────────────┐                                                       │
│   │   Retrieve   │  ← Fetch 10 docs from ChromaDB                       │
│   └──────┬───────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────┐                                                       │
│   │    Grade     │  ← LLM grades each doc for relevance                 │
│   │  Documents   │                                                       │
│   └──────┬───────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────┐     No      ┌─────────────────┐                      │
│   │  Sufficient? │────────────►│ END: No Data    │                      │
│   └──────┬───────┘             └─────────────────┘                      │
│          │ Yes                                                           │
│          ▼                                                               │
│   ┌──────────────┐                                                       │
│   │   Generate   │  ← Create answer from relevant docs                  │
│   │    Answer    │                                                       │
│   └──────┬───────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────┐     Yes     ┌─────────────────┐                      │
│   │   Validate   │────────────►│ END: Success    │                      │
│   │   Answer     │             └─────────────────┘                      │
│   └──────┬───────┘                                                       │
│          │ No (not grounded)                                             │
│          │                                                               │
│          ▼                                                               │
│   ┌──────────────┐                                                       │
│   │  Retry < 3?  │                                                       │
│   └──────┬───────┘                                                       │
│          │                                                               │
│    Yes   │   No                                                          │
│    ┌─────┴────────────────────┐                                         │
│    ▼                          ▼                                         │
│ ┌──────────────┐      ┌─────────────────┐                               │
│ │  Regenerate  │      │ END: Failed     │                               │
│ │    Answer    │──┐   │ (with warning)  │                               │
│ └──────────────┘  │   └─────────────────┘                               │
│                   │                                                      │
│                   └─────► Back to Validate                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Basic RAG | LangGraph RAG |
|---------|-----------|---------------|
| **Document Filtering** | ❌ Uses all retrieved | ✅ Grades & filters by relevance |
| **Hallucination Check** | ❌ None | ✅ Validates answer grounding |
| **Self-Correction** | ❌ None | ✅ Regenerates up to 3 times |
| **Insufficient Data** | ❌ Generates anyway | ✅ Explicit "no data" response |
| **Transparency** | ❌ Black box | ✅ Step-by-step execution trace |
| **Debugging** | ❌ Hard | ✅ Visual step display in UI |

---

## 🗂️ Project Structure

```
src/langgraph_rag/
├── __init__.py              # Package exports
├── state.py                 # GraphState schema
├── prompts.py               # All prompt templates
├── graph.py                 # Graph construction & run function
└── nodes/
    ├── __init__.py          # Node exports
    ├── retrieve.py          # Node: Retrieve from ChromaDB
    ├── grade.py             # Node: Grade document relevance
    ├── generate.py          # Node: Generate answer
    ├── validate.py          # Node: Validate grounding
    └── regenerate.py        # Node: Regenerate with stricter prompt

src/ui/
└── streamlit_langgraph.py   # Enhanced UI with step visualization
```

---

## 🔧 Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_retries` | 3 | Maximum regeneration attempts |
| `top_k_retrieval` | 10 | Documents to retrieve initially |
| `min_relevant_docs` | 1 | Minimum relevant docs to proceed |

---

## 🚀 Quick Start

### 1. Switch to Branch
```bash
git checkout rag-langgraph
```

### 2. Install Dependencies
```bash
poetry install
```

### 3. Build Vector Store (if not done)
```bash
poetry run python -m src.data_preparation.process_pipeline
poetry run python -m src.embeddings.build_vector_store
```

### 4. Run LangGraph App
```bash
poetry run streamlit run src/ui/streamlit_langgraph.py
```

### 5. Open Browser
Navigate to `http://localhost:8501`

---

## 📖 Node Descriptions

### 1. `retrieve_node`
**Purpose:** Fetch documents from ChromaDB vector store
- **Input:** `question`
- **Output:** `retrieved_documents` (10 docs)
- **Tool:** ChromaDB similarity search

### 2. `grade_documents_node`
**Purpose:** Grade each document for relevance to the question
- **Input:** `question`, `retrieved_documents`
- **Output:** `relevant_documents` (filtered)
- **Tool:** LLM with structured output (yes/no)

### 3. `generate_answer_node`
**Purpose:** Create answer using only relevant documents
- **Input:** `question`, `relevant_documents`
- **Output:** `generation`
- **Tool:** LLM with RAG prompt

### 4. `validate_answer_node`
**Purpose:** Check if answer is grounded in documents (hallucination detection)
- **Input:** `generation`, `relevant_documents`
- **Output:** `is_grounded` (boolean)
- **Tool:** LLM with structured output

### 5. `regenerate_answer_node`
**Purpose:** Retry generation with stricter grounding instructions
- **Input:** `question`, `relevant_documents`, `retry_count`
- **Output:** `generation`, `retry_count + 1`
- **Tool:** LLM with stricter prompt

---

## 🔄 Execution Scenarios

### Scenario 1: ✅ Successful Answer
```
User: "What payment methods do you accept?"

1. RETRIEVE: 10 documents fetched
2. GRADE: 7/10 documents relevant
3. CHECK: 7 ≥ 1 → proceed
4. GENERATE: Answer created
5. VALIDATE: Answer grounded ✓
6. END: Return answer with sources
```

### Scenario 2: ⚠️ Insufficient Data
```
User: "What is the weather today?"

1. RETRIEVE: 10 documents fetched
2. GRADE: 0/10 documents relevant
3. CHECK: 0 < 1 → insufficient
4. END: "I don't have information about..."
```

### Scenario 3: 🔄 Self-Correction
```
User: "What are roaming charges for USA?"

1. RETRIEVE: 10 documents fetched
2. GRADE: 5/10 documents relevant
3. GENERATE: Answer (contains hallucination)
4. VALIDATE: Not grounded ✗
5. REGENERATE: Retry 1/3
6. VALIDATE: Still not grounded ✗
7. REGENERATE: Retry 2/3
8. VALIDATE: Answer grounded ✓
9. END: Return corrected answer
```

---

## 🛠️ API Usage

### Programmatic Usage
```python
from src.langgraph_rag import run_rag_graph

# Run the graph
result = run_rag_graph(
    question="What payment methods do you accept?",
    max_retries=3
)

# Access results
print(f"Status: {result['status']}")
print(f"Answer: {result['final_answer']}")
print(f"Sources: {result['sources']}")
print(f"Steps: {len(result['steps'])}")
```

### Result Structure
```python
{
    "question": "...",
    "final_answer": "...",
    "status": "success" | "insufficient_data" | "validation_failed",
    "sources": ["billing_policy.txt", ...],
    "steps": [
        {"node": "retrieve", "status": "completed", ...},
        {"node": "grade_documents", "status": "completed", ...},
        ...
    ],
    "is_grounded": True,
    "retry_count": 0
}
```

---

## 🧪 Testing

### Run Graph Directly
```bash
poetry run python src/langgraph_rag/graph.py
```

### Run Tests
```bash
poetry run pytest tests/ -v
```

---

## 📈 Comparison: Basic vs LangGraph RAG

| Aspect | Basic RAG | LangGraph RAG |
|--------|-----------|---------------|
| **Architecture** | Linear pipeline | Stateful graph with cycles |
| **Document Handling** | Use all retrieved | Grade & filter |
| **Answer Quality** | Unvalidated | Validated for grounding |
| **Error Handling** | Generic | Specific (insufficient/failed) |
| **Retry Logic** | None | Up to 3 regenerations |
| **Observability** | Minimal | Full execution trace |
| **LLM Calls** | 1 (generation) | 3-8 (grade + generate + validate) |

---

## ⚠️ Trade-offs

### Advantages
✅ Higher answer quality  
✅ Reduced hallucinations  
✅ Better handling of edge cases  
✅ Full transparency and debugging  
✅ Graceful degradation  

### Considerations
⚠️ More LLM calls (higher cost)  
⚠️ Increased latency (~3-5x slower)  
⚠️ More complex codebase  

---

## 🔮 Future Enhancements

1. **Query Rewriting** - Improve retrieval with query expansion
2. **Parallel Grading** - Grade documents concurrently
3. **Streaming** - Stream answer generation
4. **Caching** - Cache grading results
5. **Human-in-the-Loop** - Allow human approval before final answer

---

## 📦 Dependencies Added

```toml
langgraph = "^1.0.5"
```

---

## 👤 Author

**Abhishek Roy**  
**Branch:** rag-langgraph  
**Date:** January 2026

---

## 🔗 Links

- **Main Branch:** [customer-support-rag](https://github.com/aroy1856/customer-support-rag)
- **This Branch:** [rag-langgraph](https://github.com/aroy1856/customer-support-rag/tree/rag-langgraph)
- **LangGraph Docs:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
