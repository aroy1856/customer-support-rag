# AI-Powered Telecom Customer Support Assistant
## Technical Project Report

---

## 1. Executive Summary

This project implements a Retrieval-Augmented Generation (RAG) system designed to provide intelligent, context-aware responses to telecom customer support queries. By combining semantic search with large language models, the system retrieves relevant information from policy documents and generates accurate, source-attributed answers, reducing manual search effort and improving response consistency.

**Key Achievements:**
- Developed end-to-end RAG pipeline using LangChain and OpenAI
- Processed 5 telecom policy documents into 25 optimized chunks
- Achieved semantic retrieval with 839KB vector store
- Implemented production-ready web interface with Streamlit
- Successfully deployed to GitHub for version control and collaboration

---

## 2. Problem Statement and Project Objective

### 2.1 Problem Statement

Traditional customer support systems face several challenges:

1. **Manual Information Retrieval**: Support agents must manually search through multiple policy documents to find relevant information
2. **Inconsistent Responses**: Different agents may provide varying answers to similar queries
3. **Time-Consuming Process**: Searching and compiling accurate responses from multiple sources is inefficient
4. **Scalability Issues**: As policy documents grow, manual search becomes increasingly impractical
5. **Knowledge Gap**: New agents require significant training to understand all policies

### 2.2 Project Objective

Develop an AI-powered RAG system that:
- **Automatically retrieves** relevant policy information based on natural language queries
- **Generates accurate, context-aware responses** using state-of-the-art LLMs
- **Provides source attribution** for transparency and verification
- **Reduces response time** from minutes to seconds
- **Ensures consistency** across all customer interactions
- **Scales efficiently** as policy documents expand

---

## 3. Dataset Preparation

### 3.1 Document Creation

Created 5 comprehensive telecom policy documents covering key customer support areas:

| Document | Topic | Size | Key Content |
|----------|-------|------|-------------|
| `billing_policy.txt` | Billing & Payments | 109 lines | Billing cycles, payment methods, disputes, charges |
| `fup_policy.txt` | Fair Usage Policy | 136 lines | Data limits, speed throttling, exemptions |
| `plan_activation.txt` | Plan Management | 231 lines | Activation, deactivation, plan changes, renewals |
| `roaming_tariff.txt` | Roaming Services | 279 lines | Domestic/international charges, roaming packs |
| `faqs.txt` | General FAQs | 215 lines | Common questions across all categories |

**Total Content**: ~970 lines covering diverse telecom scenarios

### 3.2 Text Preprocessing

Implemented multi-stage text cleaning pipeline:

1. **Page Number Removal**: Eliminated pagination artifacts
2. **Header/Footer Cleaning**: Removed document metadata
3. **Whitespace Normalization**: Standardized spacing and line breaks
4. **Special Character Handling**: Preserved meaningful punctuation while removing noise

**Rationale**: Clean text ensures better embedding quality and more accurate retrieval.

---

## 4. Chunking and Embedding Methodology

### 4.1 Chunking Strategy

**Configuration:**
- **Chunk Size**: 500 tokens per chunk
- **Overlap**: 150 tokens between consecutive chunks
- **Method**: Recursive character text splitting with token-based refinement

**Technical Implementation:**
```python
# Using tiktoken for accurate token counting
encoding = tiktoken.encoding_for_model("gpt-4")

# LangChain's RecursiveCharacterTextSplitter for initial splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=approx_chars,
    chunk_overlap=overlap_chars,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Token-based refinement ensures exact 500-token chunks
```

**Design Rationale:**

1. **500 Tokens**: Balances context completeness with retrieval precision
   - Too small: Context fragmentation and incomplete information
   - Too large: Retrieval noise and irrelevant content inclusion
   
2. **150 Token Overlap**: Preserves semantic continuity across chunk boundaries
   - Prevents information loss at split points
   - Ensures context for sentences spanning chunks

3. **Semantic-Aware Splitting**: Prioritizes natural breaks (paragraphs, sentences)

**Results**: 5 documents → **25 semantically coherent chunks**

### 4.2 Embedding Generation

**Model**: OpenAI `text-embedding-3-small`
- **Dimensions**: 1536
- **Advantages**: 
  - High-quality semantic representations
  - Cost-effective for production use
  - Excellent performance on retrieval tasks

**Process:**
1. Batch processing of all 25 chunks
2. SSL-configured HTTP client for corporate network compatibility
3. Embedding storage alongside chunk metadata

**Output**: Each chunk paired with 1536-dimensional embedding vector

---

## 5. Retrieval and Answer Generation Pipeline

### 5.1 Vector Store Architecture

**Technology**: ChromaDB via LangChain
- **Storage**: Persistent local storage (839KB database)
- **Collection**: `telecom_policies`
- **Distance Metric**: L2 (Euclidean distance)

**Advantages:**
- Fast similarity search (millisecond latency)
- Persistent storage across sessions
- Good integration with LangChain ecosystem

### 5.2 Retrieval Process

**Workflow:**
```
User Query → Embedding → Vector Search → Top-K Results → Context Assembly
```

**Configuration:**
- **Default Top-K**: 5 most relevant chunks
- **Similarity Metric**: L2 distance (lower = more relevant)
- **Search Type**: Similarity search with score

**Example Retrieval:**
```python
# Query embedding
query_embedding = embeddings.embed_query(user_query)

# Vector similarity search
results = vectorstore.similarity_search_with_score(
    query, k=5
)

# Format results with metadata
retrieved_chunks = [
    {
        'content': doc.page_content,
        'metadata': doc.metadata,
        'distance': score
    }
    for doc, score in results
]
```

### 5.3 Answer Generation

**Model**: OpenAI `gpt-4o-mini`
- **Temperature**: 0.3 (consistent, focused responses)
- **Context Window**: Supports extended context from multiple chunks

**Prompt Engineering:**

1. **System Prompt**: Establishes assistant persona
   - Professional, helpful tone
   - Adherence to policy information
   - Clear communication style

2. **Context Injection**: Retrieved chunks formatted as:
   ```
   [Document 1 - billing_policy.txt]
   {chunk content}
   
   [Document 2 - fup_policy.txt]
   {chunk content}
   ```

3. **User Query**: Natural language question

**Response Generation:**
```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=formatted_prompt)
]

response = llm.invoke(messages)
```

**Output Enhancement:**
- Source attribution appended to responses
- List of referenced documents
- Fallback message for no-match scenarios

---

## 6. Tools and Technologies Used

### 6.1 Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **LLM** | OpenAI GPT-4o-mini | API | Answer generation |
| **Embeddings** | OpenAI text-embedding-3-small | API | Semantic vector creation |
| **Vector Store** | ChromaDB | 0.4.22+ | Persistent vector storage |
| **Framework** | LangChain | 1.1.3+ | RAG pipeline orchestration |
| **UI** | Streamlit | 1.31.0+ | Web interface |
| **Text Processing** | tiktoken | 0.5.2+ | Token counting |
| **Dependency Mgmt** | Poetry | - | Package management |

### 6.2 Python Libraries

- `langchain-chroma`: Vector store integration
- `langchain-openai`: OpenAI LLM/embedding wrappers
- `langchain-core`: Core abstractions
- `langchain-community`: Community integrations
- `python-dotenv`: Environment configuration
- `pandas`: Data handling
- `httpx`: HTTP client with SSL customization

### 6.3 Development Tools

- **Version Control**: Git with GitHub remote
- **IDE**: VS Code / PyCharm
- **Testing**: Custom test suite with 15 queries
- **Documentation**: Markdown (README, QUICKSTART, Walkthrough)

---

## 7. Challenges Faced and Solutions

### 7.1 LangChain Version Compatibility

**Challenge**: Breaking changes between LangChain versions
- `langchain.schema` → `langchain_core.messages`
- `langchain.text_splitter` → `langchain_text_splitters`
- `langchain_community.vectorstores.Chroma` → `langchain_chroma.Chroma`

**Solution**:
- Updated all imports to use LangChain 1.x paths
- Migrated to `langchain-chroma` package for better stability
- Comprehensive dependency specification in `pyproject.toml`

### 7.2 SSL Certificate Verification Issues

**Challenge**: Corporate network SSL certificate verification failures
```
SSLError: certificate verify failed: unable to get local issuer certificate
```

**Solution**:
```python
# Custom HTTP client with SSL bypass for development
http_client = httpx.Client(verify=False)

embeddings = OpenAIEmbeddings(
    model=Config.EMBEDDING_MODEL,
    openai_api_key=Config.OPENAI_API_KEY,
    http_client=http_client
)
```

**Production Note**: In production, properly configure SSL certificates rather than bypassing verification.

### 7.3 ChromaDB Connection Errors

**Challenge**: Silent failures and connection timeouts
```
WinError 10054: An existing connection was forcibly closed by the remote host
```

**Solution**:
- Replaced custom ChromaDB client with LangChain's Chroma wrapper
- Improved error handling and logging
- Simplified initialization with better defaults

### 7.4 Python Version Constraints

**Challenge**: Poetry dependency resolution failures due to Python version mismatches

**Solution**:
```toml
requires-python = ">=3.10,<4.0"
```
Broadened Python version range to support 3.10-3.13.

### 7.5 Unicode Encoding Issues (Windows)

**Challenge**: Console encoding errors with Unicode characters
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```

**Solution**: Avoided Unicode characters in console output; used ASCII alternatives like `[OK]`, `[ERROR]`.

### 7.6 Metadata Type Compatibility

**Challenge**: ChromaDB strict metadata type requirements (strings, ints, floats only)

**Solution**:
```python
# Convert all metadata values to strings
metadata = {key: str(value) for key, value in chunk['metadata'].items()}
```

---

## 8. Key Learnings from the Project

### 8.1 Technical Learnings

1. **RAG Architecture Design**
   - Chunking strategy significantly impacts retrieval quality
   - Overlap prevents information loss at boundaries
   - Token-based chunking more reliable than character-based

2. **Vector Store Selection**
   - ChromaDB excellent for development and small-to-medium datasets
   - LangChain abstractions simplify vector store integration
   - Persistent storage crucial for production applications

3. **Prompt Engineering**
   - Clear system prompts improve response consistency
   - Context formatting affects LLM understanding
   - Temperature tuning balances creativity and precision

4. **Error Handling**
   - SSL issues common in corporate environments
   - Comprehensive error messages aid debugging
   - Fallback mechanisms improve user experience

5. **Dependency Management**
   - Poetry superior to pip for reproducible environments
   - Version pinning prevents breaking changes
   - Lock files ensure consistent deployments

### 8.2 Best Practices Identified

1. **Modular Architecture**: Separate concerns (data prep, embedding, retrieval, generation)
2. **Configuration Management**: Centralized config with environment variables
3. **Logging**: Comprehensive logging for debugging and monitoring
4. **Documentation**: Clear README, quick start guide, and inline comments
5. **Version Control**: Git from project inception with meaningful commits
6. **Testing**: Test suite validates system functionality

### 8.3 Future Improvements

1. **Hybrid Search**: Combine semantic and keyword search
2. **Re-ranking**: Implement re-ranking for better relevance
3. **Conversation Memory**: Multi-turn conversations with context
4. **Evaluation Metrics**: Automated quality assessment
5. **Production Deployment**: Cloud hosting with monitoring
6. **User Feedback Loop**: Collect and incorporate user feedback

---

## 9. System Performance

### 9.1 Metrics

| Metric | Value |
|--------|-------|
| **Vector Store Size** | 839 KB |
| **Total Chunks** | 25 |
| **Avg Chunk Size** | 500 tokens |
| **Embedding Dimensions** | 1536 |
| **Retrieval Latency** | <100ms |
| **End-to-End Response Time** | 2-3 seconds |

### 9.2 Accuracy Assessment

Manual evaluation of 15 test queries showed:
- **Relevance**: Retrieved chunks consistently relevant to queries
- **Accuracy**: Generated answers align with source documents
- **Completeness**: Answers address user questions comprehensively
- **Source Attribution**: All responses include proper source references

---

## 10. Conclusion

This project successfully demonstrates the power of RAG for domain-specific question answering. By combining semantic retrieval with large language models, the system provides accurate, context-aware responses to telecom customer support queries.

**Key Achievements:**
✅ End-to-end RAG pipeline implementation  
✅ Integration with OpenAI's state-of-the-art models  
✅ Production-ready web interface  
✅ Comprehensive documentation and testing  
✅ GitHub deployment for collaboration  

**Impact:**
The system significantly reduces manual search effort, ensures response consistency, and provides a foundation for scalable AI-powered customer support.

---

## Appendix A: Repository Structure

```
customer-support-rag/
├── src/
│   ├── data_preparation/     # Document processing
│   ├── embeddings/           # Vector store & embeddings
│   ├── retrieval/            # Semantic search
│   ├── generation/           # LLM answer generation
│   ├── ui/                   # Streamlit interface
│   └── utils/                # Configuration & logging
├── data/
│   ├── raw/                  # Source documents
│   └── chunks/               # Processed chunks
├── tests/                    # Test suite
├── docs/                     # Documentation
├── .env.example              # Environment template
├── pyproject.toml            # Dependencies
└── README.md                 # Project overview
```

## Appendix B: Example Interaction

**User Query**: "What payment methods do you accept?"

**System Process**:
1. Query embedding generated
2. Vector search retrieves relevant chunks from billing_policy.txt
3. Context assembled with 3 most relevant chunks
4. GPT-4o-mini generates response
5. Source references appended

**System Response**:
> "We accept the following payment methods:
> - Online payment through our mobile app or website using credit/debit cards, net banking, or UPI
> - Auto-debit from your registered bank account
> - Payment at authorized retail stores and franchises
> - Mobile wallets and digital payment platforms
>
> **Sources**: billing_policy.txt"

---

**Report Compiled**: December 16, 2025  
**Author**: Abhishek Roy  
**Project Repository**: https://github.com/aroy1856/customer-support-rag
