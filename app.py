import streamlit as st
from langchain_community.document_loaders import TextLoader
import logging
import nest_asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import create_retriever_tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import Union, Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from functools import wraps
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Ensure streamlit logger also shows INFO
logging.getLogger('streamlit').setLevel(logging.INFO)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a StreamHandler to show logs in Streamlit
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def log_time_info(start_time: float, step: str):
    """Log time taken for a processing step with performance metrics"""
    elapsed = time.time() - start_time
    logger.info(f"Performance: {step} completed in {elapsed:.2f}s")


# Constants for local data loading
DATA_PATH = "./data"

GRADE_PROMPT = (
    "You are an expert relevance evaluator for a retrieval-augmented generation (RAG) system.\n\n"
    "Retrieved Document:\n{context}\n\n"
    "User Query: {question}\n\n"
    "Task: Assess whether the retrieved document contains information relevant to answering the user's query.\n"
    "Consider:\n"
    "- Direct relevance: Does the document explicitly address the query?\n"
    "- Contextual relevance: Does it provide background information that helps answer the query?\n"
    "- Semantic relevance: Are there related concepts even without exact keyword matches?\n\n"
    "Respond with exactly one word: 'RELEVANT' or 'NOT_RELEVANT'"
)

REWRITE_PROMPT = (
    "You are a query optimization specialist for a semantic search system.\n\n"
    "Original Query: {question}\n\n"
    "Context: The initial retrieval did not yield sufficiently relevant results.\n\n"
    "Task: Reformulate this query to improve retrieval effectiveness.\n"
    "Strategy:\n"
    "1. Identify the core information need\n"
    "2. Expand with synonyms and related concepts\n"
    "3. Make the query more specific and targeted\n"
    "4. Use vocabulary likely to appear in source documents\n\n"
    "Reformulated Query:"
)

ANSWER_PROMPT = (
    "You are an AI assistant in a retrieval-augmented generation (RAG) system.\n\n"
    "User Query: {question}\n\n"
    "Retrieved Context:\n{context}\n\n"
    "Instructions:\n"
    "- Base your answer strictly on the provided context\n"
    "- Cite specific details from the context when relevant\n"
    "- If the context is insufficient, explicitly state the limitations\n"
    "- Maintain accuracy and avoid speculation\n"
    "- Provide clear, well-structured responses\n\n"
    "Response:"
)

RETRIEVER_CONFIG = {
    "search_type": "similarity",
    "search_kwargs": {"k": 5},  # Increased from 4 to 5 for better coverage
}

MAX_QUERY_REWRITES = 2  # Limit the number of query rewrites to prevent infinite loops


class RelevanceDecision(str, Enum):
    """Enum for relevance decision types."""

    RELEVANT = "RELEVANT"
    NOT_RELEVANT = "NOT_RELEVANT"


class RelevanceBasedAction(str, Enum):
    """Enum for actions based on relevance decision."""

    GENERATE_ANSWER = "GENERATE_ANSWER"
    REWRITE_QUESTION = "REWRITE_QUESTION"


class RelevanceCheckOutput(BaseModel):
    """
    Pydantic model for the structured output of the relevance checking LLM.
    """

    relevance: RelevanceDecision = Field(
        ...,
        description="The assessment of whether the document chunk is relevant to the user's query. Must be 'RELEVANT' or 'NOT_RELEVANT'.",
    )


# Helper function to package messages for LangGraph state updates
def _package_message_for_state(message: BaseMessage) -> Dict[str, List[BaseMessage]]:
    """
    Encapsulates a single message into the dictionary format expected by LangGraph
    when using `add_messages` reducer for a 'messages' key in the state.
    """
    return {"messages": [message]}


def check_relevance_and_suggest_action(state: MessagesState) -> str:
    """
    Check the relevance of a document chunk to a user query using a language model.
    Implements the relevance evaluation step in the agentic workflow.
    
    Args:
        state (MessagesState): Current conversation state containing messages
    Returns:
        str: Next node to execute (either GENERATE_ANSWER or REWRITE_QUESTION)
    """
    try:
        query = state["messages"][0].content
        document = state["messages"][-1].content
        logger.info(f"Evaluating relevance for query: {query[:100]}...")

        prompt = GRADE_PROMPT.format(
            context=document,
            question=query,
        )

        # Call the relevance check model
        response = st.session_state.relevance_check_model.with_structured_output(
            RelevanceCheckOutput
        ).invoke(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        decision = (
            RelevanceBasedAction.GENERATE_ANSWER.value
            if response.relevance == RelevanceDecision.RELEVANT
            else RelevanceBasedAction.REWRITE_QUESTION.value
        )
        logger.info(f"Relevance decision: {decision}")
        return decision

    except Exception as e:
        logger.error(f"Relevance check failed: {str(e)}", exc_info=True)
        st.error(f"Error during relevance check: {str(e)}")
        # Default to generating an answer on error
        return RelevanceBasedAction.GENERATE_ANSWER.value


def rewrite_query(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """
    Rewrite the user query to improve retrieval effectiveness.
    Tracks the number of rewrites to prevent infinite loops.
    Implements the adaptive query reformulation strategy.
    
    Args:
        state (MessagesState): The current state of the conversation.
    
    Returns:
        Dict[str, List[BaseMessage]]: State update with the rewritten query as a HumanMessage.
    """
    logger.info("Attempting query rewrite")
    try:
        if not state["messages"]:
            logger.error("Cannot rewrite query: MessagesState is empty.")
            return _package_message_for_state(
                HumanMessage(content="Error: No messages found to rewrite the query.")
            )

        # Count how many times we've rewritten this query
        rewrite_count = sum(1 for msg in state["messages"] if isinstance(msg, HumanMessage)) - 1
        
        if rewrite_count >= MAX_QUERY_REWRITES:
            logger.warning(f"Maximum query rewrites ({MAX_QUERY_REWRITES}) reached. Using original query.")
            # Return the original query to prevent infinite loops
            original_query = state["messages"][0].content
            return _package_message_for_state(HumanMessage(content=original_query))

        query = state["messages"][0].content if state["messages"] else "No query found"
        logger.info(f"Rewriting query (attempt {rewrite_count + 1}/{MAX_QUERY_REWRITES}): {query[:100]}...")
        
        prompt = REWRITE_PROMPT.format(question=query)
        response = st.session_state.response_model.invoke(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        
        if isinstance(response, AIMessage):
            rewritten_query = response.content
            logger.info(f"Rewritten query: {rewritten_query[:100]}...")
            return _package_message_for_state(HumanMessage(content=rewritten_query))
        else:
            raise ValueError("Unexpected response format from query rewriting model.")
            
    except Exception as e:
        logger.error(f"Error during query rewriting: {str(e)}", exc_info=True)
        return _package_message_for_state(
            HumanMessage(
                content=f"An error occurred while rewriting the query: {str(e)}"
            )
        )


def generate_answer(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """
    Generate an answer based on the retrieved document and user query.
    Implements the final answer generation step in the RAG pipeline.
    
    Args:
        state (MessagesState): Current conversation state containing messages
    
    Returns:
        Dict[str, List[BaseMessage]]: State update with generated answer as AIMessage
    """
    logger.info("Generating answer from retrieved context")
    try:
        if not state["messages"]:
            logger.error("Cannot generate answer: MessagesState is empty.")
            return _package_message_for_state(
                AIMessage(
                    content="Error: No messages found to generate an answer from."
                )
            )

        # Extract the user's original query and the retrieved document
        query = state["messages"][0].content if state["messages"] else "No query found"
        document = (
            state["messages"][-1].content
            if len(state["messages"]) > 1
            else "No document found"
        )
        
        logger.info(f"Generating answer for query: {query[:100]}...")
        
        # Use the improved answer prompt
        prompt = ANSWER_PROMPT.format(
            question=query,
            context=document
        )
        
        response = st.session_state.response_model.invoke(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        
        if isinstance(response, AIMessage):
            logger.info(f"Answer generated successfully: {response.content[:100]}...")
            return _package_message_for_state(response)
        else:
            raise ValueError("Unexpected response format from answer generation model.")
            
    except Exception as e:
        logger.error(f"Error during answer generation: {str(e)}", exc_info=True)
        return _package_message_for_state(
            AIMessage(
                content=f"I apologize, but I encountered an error while generating the answer: {str(e)}"
            )
        )


# Page configuration
st.set_page_config(
    page_title="Agentic RAG System | LangGraph Implementation",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)

nest_asyncio.apply()  # For async operations in Streamlit environment

@st.cache_resource(
    show_spinner="Loading resources...",
    ttl=7 * 24 * 60 * 60)
def load_and_prepare_resources(_data_path: str) -> Optional[Dict[str, Any]]:
    """
    Initialize all resources needed for the RAG application.
    Loads documents from the specified local data path and creates an in-memory vector database.
    This function is cached by Streamlit to avoid reprocessing on every app rerun.
    
    Args:
        _data_path: Path to directory containing text files
        
    Returns:
        Dictionary containing all initialized resources or None on failure
    """
    logger.info("="*80)
    logger.info("SYSTEM INITIALIZATION STARTING")
    logger.info(f"Data source: {_data_path}")
    logger.info("="*80)
    
    try:
        start_time = time.time()
        
        # Load all text files from the data directory in parallel
        docs = []
        text_files = [
            os.path.join(_data_path, f) for f in sorted(os.listdir(_data_path))
            if f.endswith((".txt", ".md"))
        ]
        logger.info(f"Found {len(text_files)} text files to process")
        
        def load_single_document(file_path: str) -> List[Any]:
            try:
                logger.info(f"Loading: {os.path.basename(file_path)}")
                loader = TextLoader(file_path)
                return loader.load()
            except Exception as e:
                logger.error(f"Error loading {os.path.basename(file_path)}: {str(e)}")
                return []

        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(load_single_document, f): f for f in text_files}
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    doc_content = future.result()
                    docs.extend(doc_content)
                    logger.info(f"Loaded {os.path.basename(file_path)}: {len(doc_content)} documents")
                except Exception as e:
                    logger.error(f"Failed to load {os.path.basename(file_path)}: {str(e)}")
            
        log_time_info(start_time, "document loading")

        if not docs:
            logger.error(f"No documents loaded from path: {_data_path}")
            return None
            
        logger.info(f"Successfully loaded {len(docs)} total documents")

        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        log_time_info(start_time, "document splitting")
        logger.info(f"Created {len(doc_splits)} document chunks")

        # Setup retriever with in-memory vector store
        logger.info("Setting up in-memory vector database...")
        retriever_components = setup_retriever(doc_splits)
        if not retriever_components:
            logger.error("Failed to setup retriever components")
            return None
        retriever, retriever_tool, vector_store = retriever_components

        # Initialize chat models
        logger.info("Initializing language models...")
        response_model = init_chat_model(
            model="gemini-2.0-flash",
            model_provider="google-genai",
            google_api_key=st.secrets["google_api_key"],
        )
        logger.info("Main response model initialized")

        relevance_check_model = init_chat_model(
            model="gemini-2.0-flash",
            model_provider="google-genai",
            google_api_key=st.secrets["google_api_key"],
        )
        logger.info("Relevance check model initialized")

        # Create the workflow graph
        logger.info("Building agentic workflow graph...")
        workflow = StateGraph(MessagesState)

        # Define the nodes
        workflow.add_node(generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([retriever_tool]))
        workflow.add_node(rewrite_query)
        workflow.add_node(generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "retrieve",
            check_relevance_and_suggest_action,
            {
                RelevanceBasedAction.GENERATE_ANSWER.value: "generate_answer",
                RelevanceBasedAction.REWRITE_QUESTION.value: "rewrite_query",
            },
        )

        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_query", "generate_query_or_respond")

        # Compile the graph
        graph = workflow.compile()
        logger.info("Workflow graph compiled successfully")

        log_time_info(start_time, "total resource initialization")
        logger.info("="*80)
        logger.info("SYSTEM INITIALIZATION COMPLETED")
        logger.info("="*80)
        
        return {
            "docs": docs,
            "text_splitter": text_splitter,
            "doc_splits": doc_splits,
            "retriever": retriever,
            "retriever_tool": retriever_tool,
            "vector_store": vector_store,
            "response_model": response_model,
            "relevance_check_model": relevance_check_model,
            "graph": graph,
        }
        
    except Exception as e:
        logger.error(f"Critical error during resource loading: {str(e)}", exc_info=True)
        return None


def setup_retriever(
    doc_splits: List[Document]
) -> Optional[Tuple[BaseRetriever, Any, InMemoryVectorStore]]:
    """
    Setup in-memory vector store and retriever tool.
    Returns the retriever, retriever tool, and vector store.
    """
    logger.info(f"Setting up in-memory vector store with {len(doc_splits)} document splits")
    if not doc_splits:
        logger.warning("No document splits provided to setup_retriever.")
        return None

    try:
        start_time = time.time()
        
        # Initialize embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document",
            google_api_key=st.secrets["google_api_key"]
        )
        logger.info("Embeddings model initialized.")
        
        # Create in-memory vector store from documents
        logger.info("Creating in-memory vector store and generating embeddings...")
        vector_store = InMemoryVectorStore.from_documents(
            documents=doc_splits,
            embedding=embeddings
        )
        log_time_info(start_time, "vector store creation and embedding generation")
        logger.info(f"Successfully created in-memory vector store with {len(doc_splits)} documents.")
        
        # Create retriever from vector store
        retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG["search_type"],
            search_kwargs=RETRIEVER_CONFIG["search_kwargs"]
        )
        
        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_harry_potter",
            "Search and return relevant snippets from the Harry Potter books. Use this tool when you need specific information about characters, events, spells, or locations from the series.",
        )
        logger.info("In-memory retriever setup successful.")
        return retriever, retriever_tool, vector_store
    except Exception as e:
        logger.error(f"Error setting up retriever: {str(e)}", exc_info=True)
        return None

def generate_query_or_respond(state: MessagesState):
    """
    Call the model to generate a response based on the current state.
    
    The model will decide whether to:
    - Use the retriever tool to search for information in the Harry Potter books
    - Respond directly if it's a simple question or greeting
    
    Args:
        state (MessagesState): Current conversation state containing messages
    
    Returns:
        dict: Dictionary containing updated messages with either a direct response or tool calls
    """
    # Add system instructions to guide the model's behavior
    system_message = HumanMessage(content=(
        "You are a helpful AI assistant specializing in the Harry Potter series. "
        "When users ask questions about Harry Potter (characters, events, spells, locations, etc.), "
        "use the 'retrieve_harry_potter' tool to search the books for accurate information. "
        "For greetings or simple questions unrelated to Harry Potter content, you can respond directly. "
        "Always be friendly and informative."
    ))
    
    # Combine system message with user messages
    messages = [system_message] + state["messages"]
    
    return _package_message_for_state(
        st.session_state.response_model.bind_tools(
            [st.session_state.retriever_tool]
        ).invoke(messages)
    )


def main():
    logger.info("Application starting")
    
    # Header
    st.title("Agentic RAG System")
    st.markdown("**LangGraph-Powered Retrieval-Augmented Generation with Adaptive Query Refinement**")
    
    # Technical Overview
    with st.expander("Technical Architecture", expanded=False):
        st.markdown("""
        ### System Architecture
        
        This application demonstrates advanced RAG (Retrieval-Augmented Generation) implementation using:
        
        #### **Core Technologies**
        - **LangGraph**: State machine orchestration for agentic workflows
        - **LangChain**: RAG pipeline and tool integration  
        - **Google Gemini 2.0 Flash**: LLM for generation and decision-making
        - **InMemoryVectorStore**: Fast semantic search with embeddings
        
        #### **Agentic Workflow Pattern**
        The system implements an adaptive workflow with autonomous decision-making:
        
        1. **Query Analysis Node**: LLM determines if retrieval is needed
        2. **Retrieval Node**: Semantic search using vector embeddings
        3. **Relevance Evaluation Node**: Structured output validation of retrieved context
        4. **Adaptive Routing**: 
           - If relevant → Generate answer
           - If not relevant → Rewrite query and retry (max 2 iterations)
        5. **Answer Generation Node**: Context-grounded response synthesis
        
        #### **Key Implementation Details**
        - **State Management**: LangGraph MessagesState for conversation tracking
        - **Conditional Edges**: Dynamic routing based on relevance scores
        - **Tool Integration**: Retriever exposed as LangChain tool
        - **Caching Strategy**: Streamlit `@st.cache_resource` for instant reloads
        - **Vector Embeddings**: Google `text-embedding-004` (768 dimensions)
        - **Chunk Strategy**: 250 tokens with 50-token overlap using tiktoken
        
        #### **Agentic Decision Making**
        - Uses structured output (Pydantic models) for deterministic routing
        - Implements retry logic with maximum iteration limits
        - Maintains full state history for debugging and transparency
        - Enables autonomous query reformulation without human intervention
        """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Data Source")
        st.code(f"Path: {DATA_PATH}", language="bash")
        
        if os.path.exists(DATA_PATH):
            files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(('.txt', '.md'))])
            st.markdown(f"**Documents**: {len(files)}")
            with st.expander("View Files"):
                for i, filename in enumerate(files, 1):
                    st.text(f"{i}. {filename}")
        
        st.markdown("---")
        st.subheader("Workflow Configuration")
        st.markdown(f"""
        - **Chunk Size**: 250 tokens
        - **Chunk Overlap**: 50 tokens
        - **Retrieval K**: {RETRIEVER_CONFIG['search_kwargs']['k']}
        - **Max Rewrites**: {MAX_QUERY_REWRITES}
        - **LLM**: Gemini 2.0 Flash
        - **Embedding**: text-embedding-004
        """)
        
        st.markdown("---")
        if st.session_state.get("system_ready", False):
            st.success("Status: Online")
            if st.button("Clear Cache & Reinitialize"):
                st.cache_resource.clear()
                st.session_state.system_ready = False
                st.rerun()
        else:
            st.warning("Status: Initializing...")
    
    # System Initialization
    if not st.session_state.get("system_ready", False):
        logger.info("System not ready. Initiating resource loading...")
        
        st.info("Initializing system: Loading documents, generating embeddings, and building vector database...")
        
        with st.spinner("Processing... (30-60 seconds for first run)"):
            all_resources = load_and_prepare_resources(DATA_PATH)

        if all_resources and "graph" in all_resources:
            st.session_state.update(all_resources)
            st.session_state.system_ready = True
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Documents", len(st.session_state.get('docs', [])))
            with col2:
                st.metric("Chunks", len(st.session_state.get('doc_splits', [])))
            with col3:
                st.metric("Vector Dim", "768")
            with col4:
                st.metric("Status", "Ready")
            
            st.success("System initialized successfully. Ready for queries.")
            logger.info("Resources loaded successfully and system_ready set to True.")
        else:
            st.error("System initialization failed. Check logs for details.")
            logger.error("Failed to load resources. System remains not ready.")
            st.session_state.system_ready = False
    else:
        logger.info("System already ready. Skipping resource loading.")
        
        # Show current metrics
        if st.session_state.get('doc_splits'):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Indexed Chunks", len(st.session_state['doc_splits']))
            with col2:
                st.metric("Source Documents", len(st.session_state.get('docs', [])))
            with col3:
                st.metric("System Status", "Online")

    # Main Query Interface
    if st.session_state.get("system_ready", False):
        st.markdown("---")
        st.subheader("Query Interface")
        
        query = st.text_input(
            "Enter your question:",
            placeholder="Example: Who betrayed Harry's parents?",
            help="The system will retrieve relevant context and generate a grounded answer"
        )
        
        if query:
            logger.info(f"Processing user query: '{query[:100]}...'")
            
            st.markdown("---")
            st.subheader("Workflow Execution Trace")
            
            final_answer = None  # Store the final LLM answer
            
            try:
                # Stream the graph execution
                for chunk in st.session_state.graph.stream(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": query,
                            }
                        ]
                    }
                ):
                    for node, update in chunk.items():
                        # Use expander for each node, collapsed by default
                        with st.expander(f"Step: {node}", expanded=False):
                            for msg in update.get("messages", []):
                                msg_type = getattr(msg, "type", "unknown").capitalize()
                                if hasattr(msg, "content") and msg.content:
                                    if msg.type == "ai":
                                        st.markdown("**AI Response:**")
                                        st.markdown(msg.content)
                                        final_answer = msg.content  # Save last AI message as final answer
                                    elif msg.type == "human":
                                        st.markdown(f"**User Query:** {msg.content}")
                                    else:
                                        st.markdown(f"**{msg_type}:** {msg.content}")
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    st.markdown("**Tool Invocation:**")
                                    for tool_call in msg.tool_calls:
                                        st.code(f"Tool: {tool_call.get('name', 'Unknown')}", language="text")
                # Output the final answer clearly, with support for both dark and light mode
                if final_answer:
                    st.markdown("---")
                    st.markdown("### ✅ Final Answer")
                    st.success(final_answer)
                st.success("Query processing completed")
            except Exception as e:
                st.error(f"Error during query processing: {str(e)}")
                logger.error(f"Query processing error: {str(e)}", exc_info=True)
    else:
        st.warning("System is initializing. Please wait...")


if __name__ == "__main__":
    main()
