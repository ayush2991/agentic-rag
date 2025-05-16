import streamlit as st
from langchain_community.document_loaders import TextLoader
import logging
import nest_asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
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

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


EMBEDDING_CONFIG = {
    "model": "models/text-embedding-004",
}

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
    page_title="Agentic RAG Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

nest_asyncio.apply()  # For WebBaseLoader in sync Streamlit environment


@st.cache_resource(ttl=24 * 60 * 60)
def load_and_prepare_resources(_data_path: str) -> Optional[Dict[str, Any]]:
    """
    Initialize all resources needed for the RAG application.
    Loads documents from the specified local data path.
    This function is cached; it creates and returns resources without modifying session state directly.
    The _data_path argument is prefixed with an underscore as a convention for cached function arguments.
    """
    logger.info("Starting resource initialization")
    try:
        logger.info(f"Attempting to load documents from path: {_data_path}")

        # Load all text files from the data directory
        docs = []
        for filename in os.listdir(_data_path):
            if filename.endswith(".txt") or filename.endswith(".md"):
                file_path = os.path.join(_data_path, filename)
                try:
                    loader = TextLoader(file_path)
                    docs.extend(loader.load())
                    logger.info(f"Successfully loaded {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
                    continue

        if not docs:
            logger.error(
                f"No documents loaded from path: {_data_path}. Ensure the directory exists and contains supported files."
            )
            return None
        logger.info(f"Successfully loaded {len(docs)} documents.")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        logger.info(f"Split documents into {len(doc_splits)} chunks.")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": "cpu"}
        )  # Or 'cuda' if you have a GPU

        retriever_components = setup_retriever(doc_splits, hf_embeddings)
        if not retriever_components:
            logger.error("Failed to setup retriever components.")
            return None
        vector_store, retriever, retriever_tool = retriever_components

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

        # Return all created resources
        return {
            "docs": docs,
            "text_splitter": text_splitter,
            "doc_splits": doc_splits,
            "vector_store": vector_store,
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
        return None


def setup_retriever(
    doc_splits: List[Any], embeddings: HuggingFaceEmbeddings
) -> Optional[
    Tuple[InMemoryVectorStore, Any, Any]
]:  # Using Any for retriever and tool for brevity
    """Setup vector store and retriever tool."""
    logger.info(f"Setting up retriever with {len(doc_splits)} document splits")
    if not doc_splits:
        logger.warning("No document splits provided to setup_retriever.")
        return None
    # Embeddings object itself is checked by type hinting, its validity by usage

    try:
        vector_store = InMemoryVectorStore.from_documents(doc_splits, embeddings)
        retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG["search_type"],
            search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
        )
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_blog_posts",  # Tool name
            "Search and return information about Lilian Weng blog posts.",  # Tool description
        )
        logger.info("Retriever setup successful.")
        return vector_store, retriever, retriever_tool
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
    st.title("Agentic RAG Application")

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
        with st.spinner("Initializing system... Please wait."):
            # Pass the data path to the resource loading function
            all_resources = load_and_prepare_resources(DATA_PATH)

            if all_resources and "graph" in all_resources:
                st.session_state.update(all_resources)
                st.session_state.system_ready = True
                st.success("✅ System initialized successfully!")
                logger.info(
                    "System initialized successfully and resources populated in session state."
                )
                # Display info about loaded docs from session state for consistency
                if "docs" in st.session_state and "doc_splits" in st.session_state:
                    st.write(
                        f"Loaded {len(st.session_state['docs'])} documents and "
                        f"{len(st.session_state['doc_splits'])} document splits."
                    )
            else:
                st.error(
                    "System initialization failed. Critical resources could not be loaded. Please check logs or try refreshing."
                )
                logger.error(
                    "System initialization failed: load_and_prepare_resources returned None or incomplete data."
                )
                st.session_state.system_ready = False
                # The app will be non-functional for queries if initialization fails.

    # Main application logic - only if system is ready
    query = st.text_input("Ask a question about the documents:")
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
