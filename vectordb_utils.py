# Standard library imports
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

# Third-party imports
import tiktoken
import streamlit as st
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as gemini_client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vector DB Constants
COLLECTION_NAME = "harry_potter_collection"
VECTOR_DIMENSION = 768
EMBEDDING_MODEL = "models/embedding-001"
BATCH_SIZE = 100
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
MAX_WORKERS = 5

# API Configuration
QDRANT_URL = "https://65cc3c79-a0f3-45d6-820e-cbc18bd3c4cb.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7YG52Wg3P9Hsmo8Lia0ONy59vkI23fXHpUY4_ugvG6A"

def initialize_clients(api_key: str = None) -> Tuple[QdrantClient, Any]:
    """Initialize vector database and embedding model clients."""
    try:
        # Get API key from secrets if not provided
        gemini_api_key = api_key or st.secrets["google_api_key"]
        
        # Initialize clients
        vector_db_client = QdrantClient(
    url="https://65cc3c79-a0f3-45d6-820e-cbc18bd3c4cb.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7YG52Wg3P9Hsmo8Lia0ONy59vkI23fXHpUY4_ugvG6A",
)
        gemini_client.configure(api_key=gemini_api_key)
        
        return vector_db_client, gemini_client
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise

def generate_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    """Generate vector embedding for input text."""
    try:
        embedding_params = {
            "model": EMBEDDING_MODEL,
            "content": text,
            "task_type": task_type,
        }
        
        # Only include title for retrieval_document task type
        if task_type == "retrieval_document":
            embedding_params["title"] = "Qdrant x Gemini"
            
        response = gemini_client.embed_content(**embedding_params)
        return response['embedding']
    except Exception as e:
        logger.error(f"Failed to generate embedding for text: {e}")
        raise

def split_into_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a list of items into smaller batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def count_text_tokens(text: str) -> int:
    """Calculate the number of tokens in a text string."""
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_single_embedding(text: str) -> Dict[str, Any]:
    """Generate embedding for a single text with retry logic."""
    return gemini_client.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
        title="Qdrant x Gemini"
    )

def process_text_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Process and generate embeddings for a batch of texts."""
    try:
        all_embeddings = []
        
        batches = split_into_batches(texts, BATCH_SIZE)
        total_batches = len(batches)
        
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for batch_idx, batch in enumerate(batches, 1):
                with tqdm(total=len(batch), desc=f"Batch {batch_idx}/{total_batches}") as pbar:
                    # Process batch in parallel
                    future_embeddings = [
                        executor.submit(generate_single_embedding, text)
                        for text in batch
                    ]
                    
                    # Collect results
                    batch_embeddings = []
                    for future in future_embeddings:
                        try:
                            result = future.result()
                            batch_embeddings.append(result)
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Failed to embed text: {str(e)}")
                            continue
                    
                    all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Completed batch {batch_idx}/{total_batches}")
        
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    except Exception as e:
        logger.error(f"Failed to batch embed texts: {e}")
        raise

async def initialize_vector_store(client: QdrantClient, name: str = COLLECTION_NAME) -> None:
    """Set up vector storage collection."""
    try:
        collections = client.get_collections().collections
        if any(collection.name == name for collection in collections):
            logger.info(f"Collection {name} already exists")
            return
            
        client.create_collection(
            name,
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION,
                distance=Distance.COSINE,
            )
        )
        logger.info(f"Created new collection: {name}")
    except Exception as e:
        logger.error(f"Failed to initialize collection: {e}")
        raise

def create_vector_store(client: QdrantClient, name: str) -> None:
    """Create a new vector storage collection."""
    try:
        collections = client.get_collections().collections
        if not any(collection.name == name for collection in collections):
            client.create_collection(
                name,
                vectors_config=VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=Distance.COSINE,
                )
            )
            logger.info(f"Created new collection: {name}")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise

def prepare_vector_points(embeddings: List[Dict], texts: List[str]) -> List[PointStruct]:
    """Create vector points for database storage."""
    try:
        return [
            PointStruct(
                id=idx,
                vector=embedding['embedding'],
                payload={"text": text}
            )
            for idx, (embedding, text) in enumerate(zip(embeddings, texts))
        ]
    except Exception as e:
        logger.error(f"Failed to create points: {e}")
        raise

def store_vector_points(client: QdrantClient, collection_name: str, points: List[PointStruct]) -> None:
    """Store vector points in the database."""
    try:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Successfully upserted {len(points)} points")
    except Exception as e:
        logger.error(f"Failed to upsert points: {e}")
        raise

def search_vector_store(client: QdrantClient, collection_name: str, query: str, limit: int = 5) -> List[Dict]:
    """Search for similar documents in vector store."""
    try:
        # Generate query embedding
        query_vector = generate_embedding(query, task_type="retrieval_query")
        
        # Search in collection
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        # Format results
        results = [{
            'text': result.payload['text'],
            'score': result.score
        } for result in search_results]
        
        logger.info(f"Found {len(results)} similar documents")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise

async def process_documents(data_path: str) -> None:
    """Main document processing pipeline."""
    try:
        logger.info("Initializing document processing")
        document_files = [
            os.path.join(data_path, f) for f in os.listdir(data_path) 
            if f.endswith((".txt", ".md"))
        ]
        
        # Initialize services
        vector_db, _ = initialize_clients()
        await initialize_vector_store(vector_db)
        
        # Process documents
        documents = load_documents(document_files)
        text_chunks = split_documents(documents)
        # In batches of 10, create embeddings and upload to vector db
        logger.info("Generating embeddings...")
        # Generate embeddings
        for i in range(0, len(text_chunks), 10):
            batch = text_chunks[i:i+10]
            embeddings = process_text_batch(batch)
            vector_points = prepare_vector_points(embeddings, batch)
            store_vector_points(vector_db, COLLECTION_NAME, vector_points)
        logger.info("All documents processed and stored in vector DB")
        
        # Example search
        query = "where does harry live?"
        results = search_vector_store(vector_db, COLLECTION_NAME, query)
        logger.info(f"Search results: {results}")

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise

def load_documents(file_paths: List[str]) -> List[Any]:
    """Load documents from files."""
    try:
        logger.info("Loading documents...")
        docs = []
        for file_path in file_paths:
            logger.info(f"Loading {os.path.basename(file_path)}...")
            loader = TextLoader(file_path)
            docs.extend(loader.load())
        logger.info(f"Successfully loaded {len(docs)} documents.")
        return docs
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def split_documents(documents: List[Any]) -> List[str]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(splits)} chunks.")
    return [split.page_content for split in splits]

if __name__ == "__main__":
    asyncio.run(process_documents("./data"))