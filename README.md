# Agentic RAG System: LangGraph Implementation

A production-grade Retrieval-Augmented Generation (RAG) system demonstrating advanced agentic AI patterns using LangGraph. Built for intelligent question-answering over the complete Harry Potter book series with autonomous decision-making, adaptive query refinement, and state-driven workflow orchestration.

## Technical Overview

This project showcases sophisticated RAG implementation with:

- **Agentic Workflow Orchestration**: LangGraph StateGraph for autonomous routing and decision-making
- **Adaptive Query Refinement**: Self-correcting system with automatic query rewriting (max 2 iterations)
- **Relevance Evaluation**: Structured output validation using Pydantic models
- **In-Memory Vector Store**: Fast semantic search with Google embeddings (768-dimensional)
- **Production-Ready Engineering**: Comprehensive caching, logging, and error handling

## Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | LangGraph | State machine for agentic workflows |
| RAG Framework | LangChain | Tool integration and document processing |
| LLM | Google Gemini 2.0 Flash | Response generation and decision-making |
| Embeddings | text-embedding-004 | Semantic vector representations |
| Vector Store | InMemoryVectorStore | Fast similarity search |
| UI | Streamlit | Interactive web interface |
| Tokenization | tiktoken | Token-aware text chunking |

## Architecture

### Agentic Workflow Pattern

```
User Query → Query Analysis → [Decision: Retrieve?]
                                       ↓
                              Semantic Retrieval (k=5)
                                       ↓
                            Relevance Evaluation (Structured Output)
                                       ↓
                        ┌──────────────┴──────────────┐
                        ↓                             ↓
                 Generate Answer              Rewrite Query
                        ↓                             ↓
                     Return                    Loop Back
                                          (Max 2 iterations)
```

### Key Nodes

1. **generate_query_or_respond**: LLM determines if retrieval is necessary
2. **retrieve**: ToolNode executes semantic search over vector store
3. **check_relevance_and_suggest_action**: Structured validation returns "GENERATE_ANSWER" or "REWRITE_QUESTION"
4. **rewrite_query**: Query reformulation with retry limits
5. **generate_answer**: Context-grounded response synthesis

### State Management

- **LangGraph MessagesState**: Maintains full conversation history
- **Conditional Edges**: Dynamic routing based on relevance evaluation
- **Tool Integration**: Retriever exposed as LangChain tool for LLM invocation

## Installation & Setup

### Prerequisites

- Python 3.9+
- Google API key for Gemini models
- 7 Harry Potter books as `.txt` files

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agentic-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**:
   
   Edit `config.toml`:
   ```toml
   google_api_key = "your-google-api-key-here"
   ```

4. **Place data files**:
   
   Add your Harry Potter `.txt` files to the `./data/` directory:
   ```
   data/
   ├── 01 Harry Potter and the Sorcerers Stone.txt
   ├── 02 Harry Potter and the Chamber of Secrets.txt
   └── ... (remaining books)
   ```

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the interface**: Navigate to `http://localhost:8501`

3. **First run**: Wait 30-60 seconds for initialization (document loading + vectorization)

4. **Subsequent runs**: <1 second (cached resources)

### Example Queries

- "Who betrayed Harry's parents?"
- "What is a Horcrux and how are they destroyed?"
- "Explain the prophecy about Harry and Voldemort"
- "What are the Deathly Hallows?"

## Implementation Details

### Data Processing Pipeline

```python
# Text Chunking
RecursiveCharacterTextSplitter (tiktoken-based)
├── Chunk Size: 250 tokens
├── Overlap: 50 tokens
└── Model: gpt-4 tokenizer

# Embeddings
GoogleGenerativeAIEmbeddings
├── Model: text-embedding-004
├── Dimensions: 768
└── Task Type: retrieval_document

# Vector Store
InMemoryVectorStore.from_documents()
└── Retrieval: k=5 most similar chunks
```

### Agentic Decision-Making

The system implements true agentic behavior through:

- **Autonomous Routing**: LLM decides if retrieval is needed (no hardcoded rules)
- **Self-Correction**: Automatic query rewriting when context is insufficient
- **Structured Evaluation**: Pydantic models for deterministic relevance checking
- **Retry Logic**: Maximum iteration limits prevent infinite loops (MAX_QUERY_REWRITES = 2)

### Performance Optimizations

**Caching Strategy**:
```python
@st.cache_resource(ttl=604800)  # 7-day TTL
def load_and_prepare_resources(data_path: str):
    # Loads docs, creates embeddings, builds vector store
    # Cached after first run for instant reloads
```

**Benefits**:
- First run: 30-60 seconds
- Subsequent runs: <1 second
- Automatic invalidation on code changes

### Core Functions
- `load_and_prepare_resources()`: Cached initialization of docs, embeddings, vector store, and workflow graph
- `setup_retriever()`: Creates InMemoryVectorStore with Google embeddings
- `generate_query_or_respond()`: Initial node - LLM decides retrieval necessity
- `check_relevance_and_suggest_action()`: Structured output returns "GENERATE_ANSWER" or "REWRITE_QUESTION"
- `rewrite_query()`: Query reformulation with iteration limits
- `generate_answer()`: Context-grounded response synthesis

## Project Structure

```
agentic-rag/
├── app.py                          # Main application
├── requirements.txt                # Python dependencies
├── config.toml                     # Configuration (API keys)
├── README.md                       # This file
├── TECHNICAL_OVERVIEW.md          # Detailed technical documentation
├── data/                           # Harry Potter books (7 .txt files)
│   ├── 01 Harry Potter and the Sorcerers Stone.txt
│   ├── 02 Harry Potter and the Chamber of Secrets.txt
│   └── ...
└── utils/
    └── logging_utils.py            # Logging configuration
```

## Key Differentiators

### What Makes This Implementation Stand Out

- **Conditional Routing**: Lambda functions in edges enable dynamic flow control
- **Tool Abstraction**: LangChain tools allow LLM to "use" the retriever as a function
- **State Immutability**: Proper state updates without side effects
- **Logging Infrastructure**: Structured logs with rotation (10MB max, 5 backups)
- **Error Resilience**: Try-except blocks with graceful degradation

## Configuration

### Key Constants (app.py)

```python
DATA_PATH = "./data"              # Document directory
MAX_QUERY_REWRITES = 2            # Retry limit for query refinement
RETRIEVER_CONFIG = {
    "search_kwargs": {"k": 5}     # Return top 5 chunks
}
```

### Prompts

- **GRADE_PROMPT**: Relevance evaluation with reasoning chain
- **REWRITE_PROMPT**: Query reformulation instructions
- **ANSWER_PROMPT**: Context-grounded response generation

## Logging

**Console**: INFO level with clean formatting  
**File**: DEBUG level with full context (`./logs/app.log`)  
**Rotation**: 10MB max size, 5 backup files

```python
[2025-01-15 14:30:22,123] [INFO] [app] System initialized successfully
[2025-01-15 14:30:25,456] [INFO] [app] Processing user query: 'Who betrayed...'
```

## For Technical Hiring Managers

This project demonstrates:

- **LangGraph Expertise**: Advanced StateGraph orchestration with conditional routing
- **RAG Implementation**: Complete pipeline from chunking to generation
- **Agentic AI Patterns**: Autonomous decision-making, self-correction, adaptive refinement
- **Production Engineering**: Caching strategies, logging infrastructure, error handling
- **Clean Code**: Professional, maintainable, well-documented codebase
- **Modern Stack**: Latest Google AI models, LangGraph, Streamlit

**See also**: `TECHNICAL_OVERVIEW.md` for comprehensive architecture details

---

**Author**: Aayush Agarwal  
**Contact**: [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)  
**License**: MIT


1. **True Agentic Behavior**: Not just chained function calls - autonomous decision-making at each node
2. **LangGraph Mastery**: Proper StateGraph usage with conditional edges and tool integration
3. **Production Engineering**: Comprehensive caching, logging, error handling
4. **Structured Outputs**: Pydantic models for type-safe LLM responses
5. **Transparent Execution**: Full workflow trace visible in UI
6. **Scalable Design**: Easy to swap LLMs, vector stores, or extend workflow

### Technical Sophistication