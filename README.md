# RAG AI Agent - Production-Ready Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** AI agent built with LangChain, ChromaDB, and Groq. This system implements a complete pipeline for question-answering with planning, retrieval, answer generation, and reflection capabilities. **Powered by Groq for ultra-fast inference!**

## Features

- **Intelligent Planning**: Decides when retrieval is needed based on question analysis
- **Vector Retrieval**: Uses ChromaDB for efficient document similarity search
- **LLM-Powered Answers**: Generates accurate answers using Groq (Llama 3.3, Mixtral, Gemma)
- **Answer Reflection**: Evaluates answer quality and relevance automatically
- **Comprehensive Logging**: Detailed logging at every pipeline stage
- **Streamlit UI**: User-friendly web interface for interactions
- **Evaluation Metrics**: BLEU, ROUGE, and semantic similarity scoring
- **Modular Architecture**: Clean, maintainable code structure

## Architecture

```
[User Question]
      
      
 
   Planner     → Decides if retrieval is needed
 
      
      
 
  Retriever    → Uses ChromaDB to fetch relevant context
 
      
      
 
   Answerer    → LLM generates final answer
 
      
      
 
   Reflector   → Validates and scores the answer
 
      
      
 [Final Answer + Reflection Score]
```

## Project Structure

```
rag-ai-agent/
 data/                    # Knowledge base (.txt / .pdf)
    sample.txt
    docs/
        reference.pdf
 src/
    main.py             # Entry point (run pipeline)
    config.py           # API keys, constants, paths
    pipeline/           # Modular pipeline components
       planner.py      # Decide retrieval necessity
       retriever.py    # ChromaDB embeddings + retrieval
       answerer.py     # LLM answering logic
       reflector.py    # Evaluate answer quality
       utils.py        # Common helper functions
    eval/
        metrics.py      # BLEU/ROUGE scoring
        logger.py       # Logging & monitoring
 ui/
    app.py              # Streamlit interface
    components/         # Optional UI submodules
 notebooks/
    demo_rag_agent.ipynb  # Demo Jupyter notebook
 tests/                  # Unit tests
 requirements.txt
 README.md
 .env                    # For API keys (ignored by git)
```

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Groq API key (get it from [https://console.groq.com](https://console.groq.com))
- pip package manager

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root:

```env
# Groq API Configuration
# Get your API key from: https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Groq Model (Default: llama-3.3-70b-versatile)
LLM_MODEL=llama-3.3-70b-versatile

# Embeddings Configuration (HuggingFace - Free, no API key required)
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=3

# LLM Configuration
TEMPERATURE=0.7
MAX_TOKENS=1000

# Logging
LOG_LEVEL=INFO
```

**Note:** Get your Groq API key from [https://console.groq.com](https://console.groq.com). Embeddings use HuggingFace (free, no API key needed).

### 4. Add Knowledge Base Documents

Place your documents in the `data/` directory:
- Text files (`.txt`)
- PDF files (`.pdf`)

Example:
```bash
data/
 sample.txt
 docs/
     reference.pdf
```

### 5. Run the Pipeline

#### Command Line

```bash
# Single question
python -m src.main "What is RAG?"

# Interactive mode
python -m src.main --interactive
```

#### Streamlit UI

```bash
streamlit run ui/app.py
```

#### Jupyter Notebook

Open `notebooks/demo_rag_agent.ipynb` and run the cells.

## Usage Examples

### Python API

```python
from src.main import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ask a question
result = pipeline.run("What is RAG?")

# Access results
answer = result['answer']['answer']
reflection = result['reflection']['verdict']
score = result['reflection']['score']

print(f"Answer: {answer}")
print(f"Quality: {reflection} (Score: {score})")
```

### Simple Answer Interface

```python
# Just get the answer string
answer = pipeline.answer("What is machine learning?")
print(answer)
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (required) | - |
| `LLM_MODEL` | Groq model name | `llama-3.3-70b-versatile` |
| `HF_EMBEDDING_MODEL` | HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Text chunk size | `800` |
| `CHUNK_OVERLAP` | Chunk overlap | `150` |
| `TOP_K` | Number of docs to retrieve | `3` |
| `TEMPERATURE` | LLM temperature | `0.7` |
| `MAX_TOKENS` | Max tokens for answer | `1000` |

**Groq Model Options:**
- `llama-3.3-70b-versatile` (Default - Best quality)
- `llama-3.1-8b-instant` (Faster, smaller)
- `mixtral-8x7b-32768` (Long context)
- `gemma2-9b-it` (Google's model)

**Note:** Embeddings use HuggingFace (free, no API key required)

### Retrieval Keywords

The planner uses keywords to decide if retrieval is needed. Default keywords:
- `what`, `how`, `explain`, `describe`, `tell me`
- `benefit`, `advantages`, `disadvantages`
- `why`, `when`, `where`, `who`
- `define`, `meaning`, `example`, `difference`, `compare`

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Or run specific tests:

```bash
pytest tests/test_retriever.py
pytest tests/test_pipeline.py
pytest tests/test_reflector.py
```

## Evaluation

The system includes several evaluation metrics:

- **BLEU Score**: N-gram overlap between reference and generated answers
- **ROUGE-L**: Longest Common Subsequence based scoring
- **ROUGE-N**: N-gram recall scoring
- **Semantic Similarity**: Word overlap and Jaccard similarity
- **LLM-as-Judge**: Optional LLM-based quality evaluation

## Logging

Logs are saved to `logs/rag_agent.log` and also printed to console. The logger tracks:
- Question processing
- Planner decisions
- Retrieval results
- Answer generation
- Reflection scores
- Pipeline timing

## Development

### Adding New Components

1. Create a new module in `src/pipeline/`
2. Implement the component logic
3. Integrate into `src/main.py`
4. Add tests in `tests/`

### Extending the Knowledge Base

Simply add new documents to `data/` directory. The retriever will automatically index them on the next run.

## Submission Information

This project was created for a technical assessment. For submission details, please see:
- **PROJECT_REPORT.md**: Detailed project description and challenges
- **SUBMISSION_GUIDE.md**: Guide for evaluating the project
- **SUBMISSION_CHECKLIST.md**: Complete checklist of requirements

### Quick Submission Package

To create a clean submission package:

```bash
python package_for_submission.py
```

This will create a `submission_package/` directory and `rag_ai_agent_submission.zip` file with all necessary files for submission.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- LangChain for the orchestration framework
- ChromaDB for vector storage
- Groq for ultra-fast LLM inference
- HuggingFace for free embeddings
- Streamlit for the UI framework

---

**Built for production-ready AI applications**