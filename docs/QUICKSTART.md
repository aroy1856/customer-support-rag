# Quick Start Guide

## Setup Steps

### 1. Install Dependencies
```bash
poetry install
```

**Note:** If you don't have Poetry installed, install it first:
```bash
pip install poetry
```

### 2. Configure API Key
Edit `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Build Vector Store
```bash
poetry run python -m src.embeddings.build_vector_store
```

Wait for the process to complete. You should see:
- Documents loaded
- Chunks created
- Embeddings generated
- Vector store built

### 4. Run the Application
```bash
poetry run streamlit run src/ui/streamlit_app.py
```

The app will open at `http://localhost:8501`

### 5. Test the System (Optional)
```bash
poetry run python -m tests.test_system
```

## Common Commands

### Rebuild Vector Store (if documents change)
```bash
poetry run python -m src.embeddings.build_vector_store --reset
```

### Process Documents Only (without building vector store)
```bash
poetry run python -m src.data_preparation.process_pipeline
```

### Run Tests
```bash
poetry run python -m tests.test_system
```

## Troubleshooting

**Issue**: "OPENAI_API_KEY not found"
- **Solution**: Make sure you've edited `.env` and added your actual API key

**Issue**: "Vector store not found"
- **Solution**: Run step 3 (Build Vector Store) before running the app

**Issue**: Application is slow
- **Solution**: This is normal for the first query. Subsequent queries are faster.

## Next Steps

1. Try the example questions in the web interface
2. Explore the retrieved document chunks (enable in sidebar)
3. Run the functional tests to see system performance
4. Check the logs in `logs/interactions.log`
