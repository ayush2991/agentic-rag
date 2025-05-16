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
from langchain_core.messages import AIMessage, BaseMessage
from typing import Union, Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum

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
    "search_kwargs": {"k": 4} # lambda_mult is for MMR, not typically similarity
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
        description="The assessment of whether the document chunk is relevant to the user's query. Must be 'RELEVANT' or 'NOT_RELEVANT'."
    )
    reason: str = Field(
        ...,
        min_length=5, # Ensure the reason is not overly terse
        max_length=200, # Keep the reason concise
        description="A brief explanation (5-200 characters) for the relevance assessment."
    )

    @field_validator('reason')
    @classmethod
    def reason_must_be_substantive(cls, value: str) -> str:
        if value.strip().lower() in ["none", "n/a", "no reason", "yes", "no"]:
            raise ValueError("Reason must be a substantive explanation.")
        return value

# Helper function to package messages for LangGraph state updates
def _package_message_for_state(message: BaseMessage) -> Dict[str, List[BaseMessage]]:
    """
    Encapsulates a single message into the dictionary format expected by LangGraph
    when using `add_messages` reducer for a 'messages' key in the state.
    """
    return {"messages": [message]}


@st.cache_data
def check_relevance_and_suggest_action(
        document: str,
        query: str,
        _relevance_check_model: Any) -> RelevanceBasedAction:
    '''
    Check the relevance of a document chunk to a user query using a language model.
    Args:
        document (str): The document chunk to check.
        query (str): The user query.
        _relevance_check_model: The language model to use for relevance checking.
    Returns:
        RelevanceBasedAction: The action to take based on the relevance check.
    '''
    try:
        # Call the relevance check model
        response = _relevance_check_model.with_structured_output(RelevanceCheckOutput).invoke(
            [{"role": "user", "content": f"Is this document relevant to the query '{query}'? {document}"}]
        )
        if isinstance(response, RelevanceCheckOutput):
            if response.relevance == RelevanceDecision.RELEVANT:
                return RelevanceBasedAction.GENERATE_ANSWER
            else:
                return RelevanceBasedAction.REWRITE_QUESTION
        else:
            raise ValueError("Unexpected response format from relevance check model.")
    except Exception as e:
        logger.error(f"Error during relevance check: {str(e)}", exc_info=True)
        return RelevanceBasedAction.GENERATE_ANSWER
    
# Note: The type hint for generate_answer should ideally be Dict[str, List[BaseMessage]]
# or a more specific TypedDict representing your LangGraph state, rather than str,
# as it needs to return a dictionary to update the graph's state.
@st.cache_data
def generate_answer(state: MessagesState) -> Dict[str, List[BaseMessage]]:
    """
    Generate an answer based on the document and user query.
    Args:
        document (str): The document chunk to use for generating the answer.
        query (str): The user query.
    Returns:
        str: The generated answer.
    """
    try:
        # Call the response model to generate an answer
        # MessagesState stores a list of BaseMessage objects. Access content via the .content attribute.
        if not state["messages"]:
            logger.error("Cannot generate answer: MessagesState is empty.")
            return _package_message_for_state(AIMessage(content="Error: No messages found to generate an answer from."))

        # Assuming the first message is the user's query and the last is the retrieved document.
        query = state["messages"][0].content if state["messages"] else "No query found"
        document = state["messages"][-1].content if len(state["messages"]) > 1 else "No document found"
        response = st.session_state.response_model.invoke(
            [{"role": "user", "content": f"Answer the question '{query}' based on this document: {document}"}]
        )
        if isinstance(response, AIMessage):
            return _package_message_for_state(response)
        else:
            raise ValueError("Unexpected response format from answer generation model.")
    except Exception as e:
        logger.error(f"Error during answer generation: {str(e)}", exc_info=True)
        # Return an error message in the correct format for MessagesState
        return _package_message_for_state(AIMessage(content=f"An error occurred while generating the answer: {str(e)}"))

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup logging and configuration
logger = logging.getLogger(__name__)
# Configure logger if not already configured (e.g., by Streamlit's root logger)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    logger.setLevel(logging.INFO) # Ensure level is set if handlers exist

nest_asyncio.apply() # For WebBaseLoader in sync Streamlit environment

@st.cache_resource
def load_and_prepare_resources(_urls: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """
    Initialize all resources needed for the RAG application.
    This function is cached; it creates and returns resources without modifying session state directly.
    The _urls argument is prefixed with an underscore as a convention for cached function arguments.
    """
    try:
        logger.info(f"Attempting to load documents from URLs: {_urls}")
        # WebBaseLoader expects a list of strings
        docs = WebBaseLoader(list(_urls)).load()
        if not docs:
            logger.error("No documents loaded from URLs.")
            return None
        logger.info(f"Successfully loaded {len(docs)} documents.")
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        logger.info(f"Split documents into {len(doc_splits)} chunks.")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_CONFIG["model"],
            google_api_key=st.secrets["google_api_key"]
        )
        
        retriever_components = setup_retriever(doc_splits, embeddings)
        if not retriever_components:
            logger.error("Failed to setup retriever components.")
            return None
        vector_store, retriever, retriever_tool = retriever_components

        # Initialize chat model with correct provider
        response_model = init_chat_model(
            model="gemini-2.0-flash",
            model_provider="google-genai", # Ensure this matches your init_chat_model capabilities
            google_api_key=st.secrets["google_api_key"]
        )
        logger.info("Main response model initialized.")

        # Initialize chat model for relevance checking
        relevance_check_model = init_chat_model(
            model="gemini-2.0-flash", # Using the same model type for simplicity
            model_provider="google-genai",
            google_api_key=st.secrets["google_api_key"]
        )
        logger.info("Relevance check model initialized.")
        
        # Return all created resources
        return {
            "docs": docs,
            "text_splitter": text_splitter,
            "doc_splits": doc_splits,
            "vector_store": vector_store,
            "retriever": retriever,
            "retriever_tool": retriever_tool,
            "response_model": response_model,
            "relevance_check_model": relevance_check_model
        }
    except Exception as e:
        logger.error(f"Initialization error during resource loading: {str(e)}", exc_info=True)
        return None

def setup_retriever(
    doc_splits: List[Any], 
    embeddings: GoogleGenerativeAIEmbeddings
) -> Optional[Tuple[InMemoryVectorStore, Any, Any]]: # Using Any for retriever and tool for brevity
    """Setup vector store and retriever tool."""
    if not doc_splits:
        logger.warning("No document splits provided to setup_retriever.")
        return None
    # Embeddings object itself is checked by type hinting, its validity by usage
        
    try:
        vector_store = InMemoryVectorStore.from_documents(doc_splits, embeddings)
        retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG["search_type"],
            search_kwargs=RETRIEVER_CONFIG["search_kwargs"]
        )
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_blog_posts", # Tool name
            "Search and return information about Lilian Weng blog posts." # Tool description
        )
        logger.info("Retriever setup successful.")
        return vector_store, retriever, retriever_tool
    except Exception as e:
        logger.error(f"Error setting up retriever: {str(e)}", exc_info=True)
        return None

def process_query(query: str) -> Optional[AIMessage]:
    """
    Process a user query using the response model.
    Assumes 'response_model' is available in st.session_state.
    """
    if "response_model" not in st.session_state:
        logger.error("CRITICAL: process_query called but response_model not in session_state.")
        st.error("System error: Response model not available. Please refresh the page.")
        return None

    try:
        # Invoke the model with the user query.
        # The response_model is expected to be a LangChain runnable (e.g., ChatModel bound with tools)
        response: BaseMessage = st.session_state.response_model.bind_tools([st.session_state.retriever_tool]).invoke(
            [{"role": "user", "content": query}]
        )
        if isinstance(response, AIMessage):
            return response
        else:
            logger.warning(f"Expected AIMessage, but got {type(response)}. Response: {response}")
            # Attempt to construct an AIMessage if possible, or handle error
            # For now, returning None if not AIMessage to signal an issue.
            st.error("Received an unexpected response format from the AI model.")
            return None
    except Exception as e:
        logger.error(f"Error during model invocation in process_query: {str(e)}", exc_info=True)
        st.error(f"Sorry, an error occurred while processing your query: {str(e)}")
        return None

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
    return _package_message_for_state(response_model.invoke(state["messages"]))

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
    
    # System Initialization Block
    if not st.session_state.get("system_ready", False):
        with st.spinner("Initializing system... Please wait."):
            # Pass URLS as a tuple to ensure hashability for @st.cache_resource
            all_resources = load_and_prepare_resources(tuple(URLS)) 
            
            if all_resources and "response_model" in all_resources and "relevance_check_model" in all_resources:
                st.session_state.update(all_resources)
                st.session_state.system_ready = True
                st.success("✅ System initialized successfully!")
                logger.info("System initialized successfully and resources populated in session state.")
                # Display info about loaded docs from session state for consistency
                if "docs" in st.session_state and "doc_splits" in st.session_state:
                    st.write(
                        f"Loaded {len(st.session_state['docs'])} documents and "
                        f"{len(st.session_state['doc_splits'])} document splits."
                    )
            else:
                st.error("System initialization failed. Critical resources could not be loaded. Please check logs or try refreshing.")
                logger.error("System initialization failed: load_and_prepare_resources returned None or incomplete data.")
                st.session_state.system_ready = False 
                # The app will be non-functional for queries if initialization fails.
    
    # Main application logic - only if system is ready
    if st.session_state.get("system_ready", False):
        query = st.text_input("Ask a question about the documents:")
        if query:
            with st.spinner("Retrieving information..."):
                try:
                    ai_response = process_query(query) # Expects AIMessage or None
                    
                    if ai_response: # ai_response is an AIMessage
                        displayed_something = False

                        # Display text content if available
                        if isinstance(ai_response.content, str) and ai_response.content.strip():
                            st.markdown(f"**AI:** {ai_response.content}")
                            displayed_something = True
                        elif isinstance(ai_response.content, list) and ai_response.content: # Handle list content (e.g. multimodal)
                            st.markdown("**AI:**")
                            for item in ai_response.content:
                                if isinstance(item, str): st.markdown(item)
                                # Add more specific handling for dict content if needed (e.g. images)
                                else: st.write(item)
                            displayed_something = True
                        
                        # Display tool call information if available
                        if ai_response.tool_calls:
                            tool_messages = []
                            for tc in ai_response.tool_calls:
                                tool_name = tc.get("name", "Unknown tool") # ToolCall is a TypedDict
                                tool_args = tc.get("args", {})
                                tool_messages.append(f"- Tool: `{tool_name}`, Arguments: `{tool_args}`")
                            
                            expander_title = "🛠️ Tool Call Details"
                            if not (isinstance(ai_response.content, str) and ai_response.content.strip()): # If no primary text content
                                st.info("The AI is using tools to process your request.")
                            with st.expander(expander_title, expanded=True):
                                for msg in tool_messages:
                                    st.markdown(msg)
                            displayed_something = True

                        if displayed_something:
                            st.success("✅ AI interaction processed.")
                        else:
                            st.warning("AI returned a response with no actionable content or tool calls.")
                            logger.warning(f"AI response is effectively empty: {ai_response}")
                    # If ai_response is None, process_query already showed an error.
                except Exception as e: # Catch any unexpected error during query processing or display
                    st.error(f"An error occurred while handling your query: {str(e)}")
                    logger.error(f"Error in query handling block: {str(e)}", exc_info=True)
    
    elif "system_ready" in st.session_state and not st.session_state.system_ready:
        # This state means initialization was attempted but failed.
        st.warning("System is not ready. Please check error messages above or try refreshing the page.")

if __name__ == "__main__":
    main()
