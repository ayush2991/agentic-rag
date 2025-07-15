# Agentic RAG Application: Harry Potter Edition 🪄

An intelligent Retrieval-Augmented Generation (RAG) application built with LangGraph and Streamlit that answers questions about the Harry Potter book series. The application uses an agentic workflow to intelligently decide when to retrieve documents, assess relevance, and rewrite queries for better results.

## 🌟 Features

- **Agentic Workflow**: Intelligent decision-making using LangGraph state machines
- **Document Relevance Assessment**: Automatically evaluates if retrieved documents are relevant to the query
- **Query Rewriting**: Improves queries when initial retrieval results are not relevant
- **Vector Search**: Uses Qdrant cloud vector database for efficient document retrieval
- **Streamlit Interface**: User-friendly web interface with step-by-step workflow visualization
- **Google Gemini Integration**: Powered by Google's Gemini 2.0 Flash model for high-quality responses

## 🏗️ Architecture

The application follows an agentic RAG pattern with the following workflow:

1. **Query Processing**: User submits a question
2. **Tool Decision**: Model decides whether to retrieve documents or respond directly
3. **Document Retrieval**: If needed, searches the Harry Potter document collection
4. **Relevance Check**: Evaluates if retrieved documents are relevant to the query
5. **Response Generation or Query Rewrite**: Either generates an answer or rewrites the query for better results

![Workflow Diagram](./agentic-rag-graph.png)

## 📋 Prerequisites

- Python 3.8+
- Google API key for Gemini models
- Qdrant Cloud account and API key
- Streamlit Secrets configuration

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agentic-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your data**:
   - Place your Harry Potter text files in the `./data/` directory
   - Supported formats: `.txt`, `.md`

4. **Configure secrets**:
   Create a `.streamlit/secrets.toml` file with your API keys:
   ```toml
   google_api_key = "your-google-api-key"
   ```

## 🔧 Configuration

The application uses several configuration constants that can be modified in `app.py`:

- `DATA_PATH`: Path to your document directory (default: `"./data"`)
- `RETRIEVER_CONFIG`: Search configuration for document retrieval
- Qdrant settings: Update the cloud URL and API key in the `setup_retriever` function

## 🎯 Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**: Open your browser to `http://localhost:8501`

3. **Ask questions**: Type questions about Harry Potter in the text input field

### Example Queries

- "Who betrayed Harry's parents?"
- "What is a Horcrux?"
- "How does Harry defeat Voldemort?"
- "What is the significance of the Deathly Hallows?"

## 🧩 Core Components

### State Management
- **MessagesState**: LangGraph state container for conversation history
- **Workflow Nodes**: Individual processing steps in the RAG pipeline

### Models and Tools
- **Response Model**: Google Gemini 2.0 Flash for answer generation
- **Relevance Check Model**: Separate model instance for document relevance assessment
- **QdrantRetriever**: Custom retriever implementation for Qdrant vector search

### Key Functions
- `generate_query_or_respond()`: Decides whether to retrieve or respond directly
- `check_relevance_and_suggest_action()`: Evaluates document relevance
- `rewrite_query()`: Improves queries for better retrieval results
- `generate_answer()`: Creates final responses based on retrieved documents

## 📁 Project Structure

```
agentic-rag/
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── config.toml                     # Configuration file
├── README.md                       # This file
├── agentic-rag-graph.png          # Workflow diagram
├── data/                           # Document storage
│   ├── 01 Harry Potter and the Sorcerers Stone.txt
│   ├── 02 Harry Potter and the Chamber of Secrets.txt
│   └── ...                        # Additional book files
└── utils/
    └── logging_utils.py            # Logging utilities
```

## 🔄 Workflow Details

### 1. Document Processing
- Loads text files from the `data/` directory
- Splits documents into chunks using `RecursiveCharacterTextSplitter`
- Creates embeddings using Google's text-embedding-004 model
- Stores in Qdrant cloud vector database

### 2. Query Processing Pipeline
- **Initial Assessment**: Model determines if query needs document retrieval
- **Retrieval**: Searches for relevant document chunks using vector similarity
- **Relevance Check**: Evaluates if retrieved documents answer the query
- **Decision Branch**: Either generate answer or rewrite query for better results

### 3. Response Generation
- Combines retrieved context with user query
- Uses Google Gemini to generate comprehensive answers
- Displays step-by-step process in Streamlit interface

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **LLM Orchestration**: LangGraph
- **Language Models**: Google Gemini 2.0 Flash
- **Vector Database**: Qdrant Cloud
- **Document Processing**: LangChain
- **Embeddings**: Google text-embedding-004

## 📊 Performance Features

- **Parallel Document Loading**: Uses ThreadPoolExecutor for efficient file processing
- **Caching**: Streamlit resource caching for faster subsequent loads
- **Logging**: Comprehensive logging with performance timing
- **Error Handling**: Robust error handling throughout the pipeline

## 🔍 Monitoring and Debugging

The application includes detailed logging for monitoring:
- Document loading performance
- Query processing steps
- Relevance check decisions
- Error tracking and debugging

Logs are displayed in both the console and Streamlit interface for real-time monitoring.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


---

*For questions or support, please open an issue in the GitHub repository.*