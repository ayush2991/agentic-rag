# Agentic RAG System - Technical Overview

## Project Description

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangGraph, demonstrating advanced agentic AI patterns including autonomous decision-making, adaptive query refinement, and state-driven workflow orchestration.

**Use Case**: Intelligent question-answering system over the complete Harry Potter book series (7 books)

---

## Technical Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | State machine for agentic workflow |
| **RAG Framework** | LangChain | Tool integration and document processing |
| **LLM** | Google Gemini 2.0 Flash | Response generation and decision-making |
| **Embeddings** | Google text-embedding-004 | Semantic vector representations (768-dim) |
| **Vector Store** | InMemoryVectorStore | Fast semantic search |
| **UI Framework** | Streamlit | Interactive web interface |
| **Tokenization** | tiktoken | Token-aware text chunking |

---

## Architecture

### System Flow

```
User Query → Query Analysis → Retrieval → Relevance Check → [Decision Point]
                                                                    ↓
                                                     ┌──────────────┴──────────────┐
                                                     ↓                             ↓
                                              Generate Answer              Rewrite Query
                                                     ↓                             ↓
                                                  Return                    Loop Back
                                                                        (Max 2 iterations)
```

### State Management

- **LangGraph MessagesState**: Maintains full conversation history
- **Structured Output**: Pydantic models for deterministic routing
- **Conditional Edges**: Dynamic workflow branching based on relevance scores

### Key Nodes

1. **generate_query_or_respond**: Initial LLM analysis - does this need retrieval?
2. **retrieve**: ToolNode executing semantic search (k=5 results)
3. **check_relevance_and_suggest_action**: Structured validation → "GENERATE_ANSWER" or "REWRITE_QUESTION"
4. **rewrite_query**: Query reformulation (max 2 attempts)
5. **generate_answer**: Context-grounded response synthesis

---

## Implementation Highlights

### Agentic Capabilities

- **Autonomous Routing**: LLM decides retrieval necessity without hardcoded rules
- **Self-Correction**: Automatic query rewriting when context is insufficient
- **Retry Logic**: Maximum iteration limits prevent infinite loops
- **Relevance Evaluation**: Structured output ensures deterministic decision-making

### Data Processing Pipeline

```python
# Document Loading
7 Harry Potter books → Document objects with metadata

# Chunking Strategy
RecursiveCharacterTextSplitter (tiktoken-based)
├── Chunk Size: 250 tokens
├── Overlap: 50 tokens
└── Total Chunks: ~X,XXX (depends on data)

# Vectorization
GoogleGenerativeAIEmbeddings
├── Model: text-embedding-004
├── Dimensions: 768
└── Task Type: retrieval_document

# Vector Store
InMemoryVectorStore.from_documents()
└── Retrieval: k=5 most similar chunks
```

### Performance Optimizations

- **Caching**: Streamlit `@st.cache_resource` with 7-day TTL
  - Embeddings and vector store cached in memory
  - Instant page reloads after initialization
  - Automatic cache invalidation on code changes

- **In-Memory Processing**: No external database dependencies
  - Fast initialization (30-60 seconds)
  - Zero latency for vector search
  - Suitable for datasets up to several million tokens

---

## Prompting Strategy

### Grade Prompt (Relevance Evaluation)
- Structured output with reasoning chain
- Binary classification: relevant vs. not relevant
- Explains decision for transparency

### Rewrite Prompt (Query Refinement)
- Analyzes why previous retrieval failed
- Reformulates with semantic variations
- Preserves user intent while expanding scope

### Answer Prompt (Response Generation)
- Context-grounded instructions
- Citation expectations
- Constraints against hallucination

---

## Code Quality

### Professional Standards

- ✅ Clean, emoji-free codebase suitable for technical review
- ✅ Comprehensive error handling with logging
- ✅ Type hints and docstrings throughout
- ✅ Modular function design for maintainability
- ✅ Configuration management via `config.toml`

### Logging Infrastructure

```python
# Structured logging with rotation
├── Console: INFO level with clean formatting
├── File: DEBUG level with full context
├── Max Size: 10MB with 5 backup files
└── Format: [timestamp] [level] [module] message
```

---

## Key Differentiators

### Why This Implementation Stands Out

1. **True Agentic Behavior**: Not just chained function calls - autonomous decision-making at each node
2. **LangGraph Mastery**: Proper use of StateGraph, conditional edges, and tool integration
3. **Production-Ready**: Caching, error handling, logging, and configuration management
4. **Transparent Workflow**: Full execution trace visible in UI for debugging
5. **Scalable Design**: Easy to swap LLMs, vector stores, or add new nodes

### Technical Sophistication

- **Structured Output**: Uses Pydantic models for type-safe LLM responses
- **Conditional Routing**: Lambda functions in edges for dynamic flow control
- **Tool Abstraction**: LangChain tools enable LLM to "use" the retriever
- **State Immutability**: Proper state updates without side effects
- **Async-Ready**: nest_asyncio integration for Streamlit compatibility

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key in config.toml
google_api_key = "your-key-here"

# Run application
streamlit run app.py
```

**First Run**: 30-60 seconds (document loading + vectorization)  
**Subsequent Runs**: <1 second (cached resources)

---

## Future Enhancements

- [ ] Multi-query retrieval for comprehensive coverage
- [ ] Hybrid search (dense + sparse embeddings)
- [ ] Persistent vector store (Qdrant/Pinecone)
- [ ] Conversation memory for follow-up questions
- [ ] Source attribution with direct text snippets
- [ ] A/B testing different prompting strategies

---

## For Hiring Managers

This project demonstrates:

✅ **LangGraph Expertise**: Advanced workflow orchestration, not just LangChain chains  
✅ **RAG Implementation**: Complete pipeline from chunking to generation  
✅ **Agentic AI Patterns**: Autonomous decision-making, self-correction, adaptive behavior  
✅ **Production Engineering**: Caching, logging, error handling, configuration management  
✅ **Clean Code**: Professional, maintainable, well-documented codebase  
✅ **Modern Stack**: Latest Google AI models, LangGraph, Streamlit

**Contact**: Aayush Agarwal | ayush2991@gmail.com
