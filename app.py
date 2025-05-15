import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
import logging
import nest_asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from typing import Union, Dict, Any

# Constants
URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

EMBEDDING_CONFIG = {
    "model": "models/text-embedding-004",
}

RETRIEVER_CONFIG = {
    "search_type": "similarity",
    "search_kwargs": {"k": 4, "lambda_mult": 0.5}
}

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup logging and configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
nest_asyncio.apply()

@st.cache_resource
def initialize_resources(urls):
    """Initialize all resources needed for the RAG application."""
    try:
        docs = WebBaseLoader(urls).load()
        if not docs:
            logger.error("No documents loaded")
            return None
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_CONFIG["model"],
            google_api_key=st.secrets["google_api_key"]
        )
        
        vector_store, retriever, retriever_tool = setup_retriever(doc_splits, embeddings)
        if not all([vector_store, retriever, retriever_tool]):
            logger.error("Failed to setup retriever components")
            return None

        # Initialize chat model with correct provider
        response_model_with_tools = init_chat_model(
            model="gemini-2.0-flash",
            model_provider="google-genai",
            google_api_key=st.secrets["google_api_key"]
        ).bind_tools([retriever_tool])
        
        # Store resources in session state
        st.session_state.update({
            "docs": docs,
            "text_splitter": text_splitter,
            "doc_splits": doc_splits,
            "vector_store": vector_store,
            "retriever": retriever,
            "retriever_tool": retriever_tool,
            "response_model": response_model_with_tools
        })
        
        return {
            "docs": docs,
            "doc_splits": doc_splits,
        }
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        return None

def is_system_initialized() -> bool:
    """Check if all required components are initialized in session state."""
    required_components = [
        "docs",
        "text_splitter",
        "doc_splits",
        "vector_store",
        "retriever",
        "retriever_tool",
        "response_model"
    ]
    return all(comp in st.session_state for comp in required_components)

def ensure_initialized():
    """Ensure system is initialized or initialize it."""
    if not is_system_initialized():
        with st.spinner("Initializing system..."):
            resources = initialize_resources(URLS)
            if not resources or not is_system_initialized():
                st.error("Failed to initialize system")
                st.stop()
            st.success("✅ System initialized successfully")
            st.write(
                f"Loaded {len(resources['docs'])} documents and "
                f"{len(resources['doc_splits'])} document splits."
            )

def process_query(query: str) -> Union[Dict[str, Any], AIMessage]:
    """Process a user query using the retriever tool."""
    ensure_initialized()
    return {"messages": [st.session_state.response_model.invoke(
        [{"role": "user", "content": query}]
    )]}

def setup_retriever(doc_splits, embeddings):
    """Setup vector store and retriever tool."""
    vector_store = InMemoryVectorStore.from_documents(doc_splits, embeddings)
    retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"]
    )
    return vector_store, retriever, create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts."
    )

def generate_query_or_respond(response_model, state: MessagesState):
    """Call the model to generate a response based on the current state.
    
    Given the question, it will decide to retrieve using the retriever tool,
    or simply respond to the user.

    Args:
        response_model: The language model with bound tools
        state (MessagesState): Current conversation state containing messages
        
    Returns:
        dict: Dictionary containing updated messages

    Examples:
        >>> # Simple question that doesn't need retrieval
        >>> state = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        >>> result = generate_query_or_respond(model, state)
        >>> # Returns: {"messages": [{"role": "assistant", "content": "4"}]}
        
        >>> # Question requiring document retrieval
        >>> state = {"messages": [{"role": "user", 
        ...          "content": "What does Lilian Weng say about reward hacking?"}]}
        >>> result = generate_query_or_respond(model, state)
        >>> # Returns: {"messages": [{"role": "assistant", 
        ...          "content": "Let me search the documents...",
        ...          "tool_calls": [{"name": "retrieve_blog_posts", ...}]}]}
    """
    return {"messages": [response_model.invoke(state["messages"])]}

def main():
    st.title("Agentic RAG Application")
    
    with st.sidebar:
        st.header("📚 Source Documents")
        for url in URLS:
            st.markdown(f"- [{url.split('/')[-2]}]({url})")

    # Show an image of flowchart this will follow
    st.write(
        "This application uses a Retrieval-Augmented Generation (RAG) approach to "
        "answer questions based on the provided documents. It retrieves relevant "
        "information and generates responses using a language model."
    )
    st.image("./agentic-rag-graph.png", caption="Flowchart of the RAG process", use_container_width=True)
    
    try:
        ensure_initialized()
        query = st.text_input("Ask a question about the documents:")
        if query:
            with st.spinner("Retrieving information..."):
                raw_response = process_query(query)
                st.write(raw_response["messages"][-1])
            st.success("✅ Information retrieved successfully")
    
    except Exception as e:
        st.error(f"System error: {str(e)}")
        logger.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()
