# AI-Powered Telecom Customer Support Assistant using RAG

An intelligent customer support system that uses Retrieval-Augmented Generation (RAG) to answer telecom-related queries about billing, plans, roaming, and policies.

## 🎯 Project Overview

This project implements a RAG-based chatbot capable of:
- Understanding customer queries in natural language
- Retrieving relevant information from telecom policy documents
- Generating accurate, source-backed answers using GPT-4o-mini
- Providing a user-friendly web interface for customer interactions

## ✨ Features

- **Intelligent Document Retrieval**: Uses semantic search with OpenAI embeddings and ChromaDB
- **Accurate Answer Generation**: Powered by GPT-4o-mini with context-aware responses
- **Source References**: Every answer includes references to source policy documents
- **Professional Web Interface**: Built with Streamlit for easy interaction
- **Comprehensive Logging**: Tracks all queries, retrieved documents, and responses
- **Functional Testing**: Includes 15+ test queries for validation

## 📁 Project Structure

```
customer-support-rag/
├── data/
│   ├── raw/                    # Original policy documents
│   ├── processed/              # Cleaned documents
│   └── chunks/                 # Chunked data with embeddings
├── src/
│   ├── data_preparation/       # Document loading, cleaning, chunking
│   ├── embeddings/             # Embedding generation and vector store
│   ├── retrieval/              # Document retrieval
│   ├── generation/             # Answer generation with LLM
│   ├── ui/                     # Streamlit web interface
│   └── utils/                  # Configuration and logging
├── tests/                      # Functional tests
├── logs/                       # Interaction logs
├── docs/                       # Documentation
└── chroma_db/                  # Vector database (created after setup)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Git (optional)

### Installation

1. **Clone or download the repository**
   ```bash
   cd customer-support-rag
   ```

2. **Install dependencies**
   
   Using Poetry (recommended):
   ```bash
   poetry install
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

### Building the Vector Store

Before using the system, you need to process the documents and build the vector store in two steps:

**Step 1: Process Documents and Generate Embeddings**

```bash
poetry run python -m src.data_preparation.process_pipeline
```

Or if using pip:
```bash
python -m src.data_preparation.process_pipeline
```

This will:
1. Load all telecom policy documents from `data/raw/`
2. Clean the text (remove headers, footers, normalize whitespace)
3. Split them into 500-token chunks with 150-token overlap
4. Generate embeddings using OpenAI for each chunk
5. Save processed chunks with embeddings to `data/chunks/chunks_with_embeddings.json`

**Step 2: Build the Vector Store**

```bash
poetry run python -m src.embeddings.build_vector_store
```

Or if using pip:
```bash
python -m src.embeddings.build_vector_store
```

This will:
1. Load the processed chunks with embeddings
2. Create a ChromaDB vector store
3. Store all chunks in the database at `chroma_db/`

**Note**: Both steps require an internet connection and will make API calls to OpenAI.

### Running the Application

Start the Streamlit web interface:

```bash
poetry run streamlit run src/ui/streamlit_app.py
```

Or if using pip:
```bash
streamlit run src/ui/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

### Web Interface

1. Open the application in your browser
2. Type your question in the text area
3. Click "Get Answer" to receive a response
4. View the answer along with source references

**Example Questions:**
- "What payment methods do you accept?"
- "How do I activate international roaming?"
- "What is Fair Usage Policy?"
- "Can I change my plan anytime?"

### Running Tests

Execute the test suite using pytest:

```bash
poetry run pytest tests/ -v
```

Or if using pip:
```bash
pytest tests/ -v
```

For coverage report:
```bash
poetry run pytest tests/ --cov=src --cov-report=html
```

This will run all tests including:
- Retriever functionality tests
- Answer generation tests
- End-to-end system tests

## 🛠️ Technical Details

### Technology Stack

- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: ChromaDB (persistent local storage)
- **Framework**: LangChain
- **UI**: Streamlit
- **Text Processing**: tiktoken for accurate token counting

### RAG Pipeline

1. **Document Processing**
   - Load 5 telecom policy documents (billing, FUP, activation, roaming, FAQs)
   - Clean text (remove headers, footers, normalize whitespace)
   - Chunk into 500-token segments with 150-token overlap

2. **Embedding Generation**
   - Generate embeddings for all chunks using OpenAI
   - Store in ChromaDB with metadata

3. **Query Processing**
   - Convert user query to embedding
   - Perform similarity search (Top-5 by default)
   - Retrieve most relevant document chunks

4. **Answer Generation**
   - Inject retrieved context into prompt
   - Generate answer using GPT-4o-mini
   - Include source references

### Configuration

Key parameters can be adjusted in `.env`:

```
CHUNK_SIZE=500              # Token size for chunks
CHUNK_OVERLAP=150           # Token overlap between chunks
TOP_K=5                     # Number of chunks to retrieve
LLM_MODEL=gpt-4o-mini      # OpenAI model to use
EMBEDDING_MODEL=text-embedding-3-small
```

## 📊 Project Tasks

This project fulfills the following requirements:

- ✅ **Task 1**: Dataset creation with 5 telecom policy documents
- ✅ **Task 2**: Text processing and chunking (500 tokens, 150 overlap)
- ✅ **Task 3**: Embedding generation and ChromaDB vector store
- ✅ **Task 4**: Top-K similarity-based retrieval
- ✅ **Task 5**: RAG answer generation with GPT-4o-mini
- ✅ **Task 6**: Streamlit web interface
- ✅ **Task 7**: System logging and functional testing
- ✅ **Task 8**: Documentation and reporting

## � Project Documentation

Detailed documentation and reports are available in the `docs/` directory:

- **[Technical Report](docs/Technical_Report.md)**: Comprehensive 6-page report covering methodology, implementation, and challenges.
- **[Business Summary](docs/Business_Summary.md)**: Executive summary focusing on business value, efficiency gains, and ROI.
- **[RAG Project Submission](docs/RAG_Project_Submission.html)**: Interactive notebook analysis and report.

## �📝 Logging

All interactions are logged to:
- `logs/interactions.log` - Standard log format
- `logs/interactions.jsonl` - JSON lines format for analysis

Each log entry includes:
- Timestamp
- User query
- Retrieved document chunks with scores
- Generated response
- Metadata (model, top_k, etc.)

## 🔧 Troubleshooting

### "OpenAI API key not found"
- Make sure you've set `OPENAI_API_KEY` in the `.env` file
- Restart the application after updating `.env`

### "Vector store not found"
- Run `python -m src.embeddings.build_vector_store` to build the vector store
- Make sure the build completed successfully

### "No relevant documents found"
- Try rephrasing your question
- Make sure your question is related to telecom policies
- Check that the vector store was built correctly

## 📄 License

This project is created for educational purposes as part of an AI/ML capstone project.

## 👥 Author

Abhishek Roy

## 🙏 Acknowledgments

- OpenAI for GPT-4o-mini and embedding models
- LangChain for RAG framework
- ChromaDB for vector storage
- Streamlit for the web interface
