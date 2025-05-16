import streamlit as st
from langchain_community.document_loaders import TextLoader
import logging
from qdrant_client import QdrantClient, models
import nest_asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
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
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def log_time_info(start_time: float, step: str):
    """Log time taken for a processing step"""
    elapsed = time.time() - start_time
    logger.info(f"⏱️ Time taken for {step}: {elapsed:.2f} seconds")

# Constants for local data loading
DATA_PATH = "./data"

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'RELEVANT' or 'NOT_RELEVANT' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate a concise, improved question:"
)

RETRIEVER_CONFIG = {
    "search_type": "similarity",
    "search_kwargs": {"k": 4},
}


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


def check_relevance_and_suggest_action(state: MessagesState) -> Dict[str, str]:
    """
    Check the relevance of a document chunk to a user query using a language model.
    Args:
        state (MessagesState): Current conversation state containing messages
    Returns:
        Dict[str, str]: Dictionary with next node to execute
    """
    try:
        query = state["messages"][0].content
        document = state["messages"][-1].content
        logger.info(f"Checking relevance for query: {query[:50]}...")

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
            RelevanceBasedAction.GENERATE_ANSWER
            if response.relevance == RelevanceDecision.RELEVANT
            else RelevanceBasedAction.REWRITE_QUESTION
        )
        logger.info(f"Relevance decision for query: {decision}")
        return decision

    except Exception as e:
        logger.error(f"Relevance check failed: {str(e)}", exc_info=True)
        st.error(f"An error occurred while checking document relevance: {str(e)}")
        return RelevanceBasedAction.GENERATE_ANSWER


def rewrite_query(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """
    Rewrite the user query based on the document and user query.
    Args:
        state (MessagesState): The current state of the conversation.
    The state should contain the user's query and the retrieved document.
    1. The first message is assumed to be the user's query.
    2. The last message is assumed to be the retrieved document.
    Returns:
        Dict[str, List[BaseMessage]]: State update with the rewritten query as a HumanMessage.
    """
    logger.debug("Attempting to rewrite query")
    try:
        if not state["messages"]:
            logger.error("Cannot rewrite query: MessagesState is empty.")
            return _package_message_for_state(
                HumanMessage(content="Error: No messages found to rewrite the query.")
            )

        query = state["messages"][0].content if state["messages"] else "No query found"
        prompt = REWRITE_PROMPT.format(
            question=query,
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
            # Convert the AI's rewritten query into a HumanMessage
            return _package_message_for_state(HumanMessage(content=response.content))
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
    Generate an answer based on the document and user query.
    Args:
        document (str): The document chunk to use for generating the answer.
        query (str): The user query.
    Returns:
        str: The generated answer.
    """
    logger.debug("Generating answer")
    try:
        # Call the response model to generate an answer
        # MessagesState stores a list of BaseMessage objects. Access content via the .content attribute.
        if not state["messages"]:
            logger.error("Cannot generate answer: MessagesState is empty.")
            return _package_message_for_state(
                AIMessage(
                    content="Error: No messages found to generate an answer from."
                )
            )

        # Assuming the first message is the user's query and the last is the retrieved document.
        query = state["messages"][0].content if state["messages"] else "No query found"
        document = (
            state["messages"][-1].content
            if len(state["messages"]) > 1
            else "No document found"
        )
        response = st.session_state.response_model.invoke(
            [
                {
                    "role": "user",
                    "content": f"Answer the question '{query}' based on this document: {document}",
                }
            ]
        )
        if isinstance(response, AIMessage):
            return _package_message_for_state(response)
        else:
            raise ValueError("Unexpected response format from answer generation model.")
    except Exception as e:
        logger.error(f"Error during answer generation: {str(e)}", exc_info=True)
        # Return an error message in the correct format for MessagesState
        return _package_message_for_state(
            AIMessage(
                content=f"An error occurred while generating the answer: {str(e)}"
            )
        )


# Page configuration
st.set_page_config(
    page_title="Agentic RAG Application: Harry Potter Edition",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

nest_asyncio.apply()  # For WebBaseLoader in sync Streamlit environment

@st.cache_resource(
    show_spinner="Loading resources...",
    ttl=7 * 24 * 60 * 60)
def load_and_prepare_resources(_data_path: str) -> Optional[Dict[str, Any]]:
    """
    Initialize all resources needed for the RAG application.
    Loads documents from the specified local data path.
    This function is cached; it creates and returns resources without modifying session state directly.
    The _data_path argument is prefixed with an underscore as a convention for cached function arguments.
    """
    logger.info("Starting resource initialization")
    logger.info(f"Executing load_and_prepare_resources for path: {_data_path}. (Cache status: if this msg repeats often on reruns without code changes, it's a cache miss or forced re-run due to initialization failure).")
    try:
        logger.info(f"Attempting to load documents from path: {_data_path}")

        # Load all text files from the data directory in parallel
        start_time = time.time()
        docs = []
        text_files = [
            os.path.join(_data_path, f) for f in os.listdir(_data_path) 
            if f.endswith((".txt", ".md"))
        ]
        logger.info(f"Found {len(text_files)} text files to process")
        
        def load_single_document(file_path: str) -> List[Any]:
            try:
                logger.info(f"Loading {os.path.basename(file_path)}...")
                loader = TextLoader(file_path)
                return loader.load()
            except Exception as e:
                logger.error(f"Error loading {os.path.basename(file_path)}: {str(e)}")
                return []

        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(load_single_document, f): f for f in text_files}
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    doc_content = future.result()
                    docs.extend(doc_content)
                    logger.info(f"✅ Successfully loaded {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {os.path.basename(file_path)}: {str(e)}")
            
        log_time_info(start_time, "document loading")

        if not docs:
            logger.error(
                f"No documents loaded from path: {_data_path}. Ensure the directory exists and contains supported files."
            )
            logger.warning("load_and_prepare_resources: No documents loaded. Returning None.")
            return None
        logger.info(f"Successfully loaded {len(docs)} documents.")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        log_time_info(start_time, "document splitting")
        logger.info(f"Split documents into {len(doc_splits)} chunks.")

        retriever_components = setup_retriever(doc_splits)
        if not retriever_components:
            logger.error("Failed to setup retriever components.")
            logger.warning("load_and_prepare_resources: setup_retriever failed. Returning None.")
            return None
        retriever, retriever_tool = retriever_components

        # Initialize chat model with correct provider
        response_model = init_chat_model(
            model="gemini-2.0-flash",
            model_provider="google-genai",  # Ensure this matches your init_chat_model capabilities
            google_api_key=st.secrets["google_api_key"],
        )
        logger.info("Main response model initialized.")

        # Initialize chat model for relevance checking
        relevance_check_model = init_chat_model(
            model="gemini-2.0-flash",  # Using the same model type for simplicity
            model_provider="google-genai",
            google_api_key=st.secrets["google_api_key"],
        )
        logger.info("Relevance check model initialized.")

        # Create a graph for the workflow
        workflow = StateGraph(MessagesState)

        # Define the nodes we will cycle between
        workflow.add_node(generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([retriever_tool]))
        workflow.add_node(rewrite_query)
        workflow.add_node(generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            # Assess LLM decision (call `retriever_tool` tool or respond to the user)
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            check_relevance_and_suggest_action,
            {
                # Map the enum values to node names
                RelevanceBasedAction.GENERATE_ANSWER: "generate_answer",
                RelevanceBasedAction.REWRITE_QUESTION: "rewrite_query",
            },
        )

        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_query", "generate_query_or_respond")

        # Compile
        graph = workflow.compile()

        logger.info("State graph built successfully.")

        logger.info("load_and_prepare_resources completed successfully. Returning resources.")
        # Return all created resources
        return {
            "docs": docs,
            "text_splitter": text_splitter,
            "doc_splits": doc_splits,
            "retriever": retriever,
            "retriever_tool": retriever_tool,
            "response_model": response_model,
            "relevance_check_model": relevance_check_model,
            "graph": graph,
        }
    except Exception as e:
        logger.error(
            f"Initialization error during resource loading: {str(e)}", exc_info=True
        )
        logger.warning("load_and_prepare_resources failed due to an exception. Returning None.")
        return None


class QdrantRetriever(BaseRetriever):
    """Custom retriever for Qdrant vector store."""
    
    client: QdrantClient
    collection_name: str
    k: int = 4
    embeddings: GoogleGenerativeAIEmbeddings

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, client: QdrantClient, collection_name: str, k: int = 4, api_key: str = None):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_query",
            google_api_key=api_key or st.secrets["google_api_key"]
        )
        # Initialize Pydantic model with all required fields
        super().__init__(
            client=client,
            collection_name=collection_name,
            k=k,
            embeddings=embeddings
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=self.k,
            with_payload=True
        )
        
        # Convert results to Documents
        return [
            Document(
                page_content=result.payload.get('text_content', ''),
                metadata={
                    'score': result.score,
                    'source_file': result.payload.get('source_file'),
                    'chunk_index': result.payload.get('chunk_index')
                }
            ) 
            for result in search_results
        ]


def setup_retriever(
    doc_splits: List[Document]
) -> Optional[Tuple[BaseRetriever, Any]]:
    """Setup Qdrant vector store and retriever tool."""
    logger.info(f"Setting up Qdrant retriever with {len(doc_splits)} document splits")
    if not doc_splits:
        logger.warning("No document splits provided to setup_retriever.")
        return None

    try:
        start_time = time.time()
        
        # Initialize Qdrant client
        QDRANT_CLOUD_URL = "https://65cc3c79-a0f3-45d6-820e-cbc18bd3c4cb.us-east4-0.gcp.cloud.qdrant.io:6333" # Recommended: Set as environment variable
        QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7YG52Wg3P9Hsmo8Lia0ONy59vkI23fXHpUY4_ugvG6A"   # Recommended: Set as environment variable

        qdrant_client = QdrantClient(
            url=QDRANT_CLOUD_URL,
            api_key=QDRANT_API_KEY
        )
        logger.info("Qdrant client initialized successfully.")
        # Initialize collection
        
        
        # Create custom retriever with proper field initialization
        retriever = QdrantRetriever(
            client=qdrant_client,
            collection_name="harry_potter_collection_2",
            api_key=st.secrets["google_api_key"]  # Pass API key explicitly
        )
        
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_harry_potter",
            "Search and return snippets from the Harry Potter books. Uses Qdrant vector store.",
        )
        logger.info("Retriever setup successful.")
        return retriever, retriever_tool
    except Exception as e:
        logger.error(f"Error setting up retriever: {str(e)}", exc_info=True)
        return None

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state.

    Given the question, it will decide to retrieve using the retriever tool,
    or simply respond to the user.

    Args:
        state (MessagesState): Current conversation state containing messages

    Returns:
        dict: Dictionary containing updated messages

    Examples:
        >>> # Simple question that doesn't need retrieval
        >>> state = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        >>> result = generate_query_or_respond(state)
        >>> # Returns: {"messages": [{"role": "assistant", "content": "4"}]}

        >>> # Question requiring document retrieval
        >>> state = {"messages": [{"role": "user",
        ...          "content": "What does Lilian Weng say about reward hacking?"}]}
        >>> result = generate_query_or_respond(state)
        >>> # Returns: {"messages": [{"role": "assistant",
        ...          "content": "Let me search the documents...",
        ...          "tool_calls": [{"name": "retrieve_blog_posts", ...}]}]}
    """
    return _package_message_for_state(
        st.session_state.response_model.bind_tools(
            [st.session_state.retriever_tool]  # Wrap tool in a list
        ).invoke(state["messages"])
    )


def main():
    logger.info("Application starting")
    st.title("Agentic RAG Application: Harry Potter Edition")

    with st.sidebar:
        st.header("📚 Source Documents")
        st.markdown(f"Loading documents from local directory: `{DATA_PATH}`")
        # Show the list of loaded documents
        st.markdown("### Loaded Documents:")
        for filename in sorted(os.listdir(DATA_PATH)):
            st.markdown(f"- {filename}")

    # Show an image of flowchart this will follow
    st.write(
        "This application uses a Retrieval-Augmented Generation (RAG) approach to "
        "answer questions based on the provided documents. It retrieves relevant "
        "information and generates responses using a language model."
    )
    st.image(
        "./agentic-rag-graph.png",
        caption="Flowchart of the RAG process",
        use_container_width=True,
    )

    # System Initialization Block
    if not st.session_state.get("system_ready", False):
        logger.info("System not ready. Attempting to load resources.")
        with st.spinner("Initializing system... Please wait."):
            # Pass the data path to the resource loading function
            all_resources = load_and_prepare_resources(DATA_PATH)

            if all_resources and "graph" in all_resources:
                st.session_state.update(all_resources)
                st.session_state.system_ready = True
                st.success("✅ System initialized successfully!")
                logger.info("Successfully loaded resources and set system_ready to True.")
                # Display info about loaded docs from session state for consistency
                if "docs" in st.session_state and "doc_splits" in st.session_state:
                    st.write(
                        f"Loaded {len(st.session_state['docs'])} documents and "
                        f"{len(st.session_state['doc_splits'])} document splits."
                    )
                else:
                    logger.warning("Docs or doc_splits not found in session_state after resource loading.")
            else:
                st.error(
                    "System initialization failed. Critical resources could not be loaded. Please check logs or try refreshing."
                )
                logger.error(
                    "Failed to load resources or graph missing. System_ready will be False."
                )
                # Ensure system_ready is explicitly False if it failed
                st.session_state.system_ready = False
                logger.error( # Duplicated from original, kept for explicitness
                    "System initialization failed: load_and_prepare_resources returned None or incomplete data."
                )
                logger.error(
                    "System initialization failed: load_and_prepare_resources returned None or incomplete data."
                )
                st.session_state.system_ready = False
                # The app will be non-functional for queries if initialization fails.

    else:
        logger.info("System already ready. Skipping resource loading.")
    # Main application logic - only if system is ready
    query = st.text_input("Ask any question to the agentic RAG application:", placeholder="\"Who betrayed Harry's parents?\" or \"How to make pasta?\"")
    if query:
        logger.info(f"Processing query: '{query[:100]}...'")
        # Create a container for the stream output
        response_container = st.container()

        with response_container:
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
                    # Create an expander for each step
                    with st.expander(f"🔄 Step: {node}", expanded=True):
                        # Display all messages in this update
                        for msg in update.get("messages", []):
                            if hasattr(msg, "content"):
                                st.markdown(f"**{msg.type}**: {msg.content}")
                            # If there are tool calls, display them
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                st.markdown("🛠️ **Tool Calls:**")
                                for tool_call in msg.tool_calls:
                                    st.code(f"Tool: {tool_call.get('name', 'Unknown')}")


if __name__ == "__main__":
    main()
