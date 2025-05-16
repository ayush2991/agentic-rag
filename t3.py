import os
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict, Any
import uuid
import getpass # To prompt for API key if not set

# --- Configuration ---
DATA_DIR = "./data/"
# --- Qdrant Cloud Specific Configuration ---
# IMPORTANT: Replace with your actual Qdrant Cloud Cluster URL and API Key
# Get these from your Qdrant Cloud dashboard (e.g., https://cloud.qdrant.io/)
QDRANT_CLOUD_URL = "https://65cc3c79-a0f3-45d6-820e-cbc18bd3c4cb.us-east4-0.gcp.cloud.qdrant.io:6333" # Recommended: Set as environment variable
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7YG52Wg3P9Hsmo8Lia0ONy59vkI23fXHpUY4_ugvG6A"   # Recommended: Set as environment variable

COLLECTION_NAME = "harry_potter_collection_2" # Changed collection name for clarity
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 16

GOOGLE_API_KEY = "AIzaSyAWbJfFtilX1JQ6dgjNmk9Z99nplDGBPkI"  # Set your Google API Key as an environment variable

# --- Gemini Embedding Model Setup ---
# Ensure GOOGLE_API_KEY environment variable is set or prompt for it
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="retrieval_document"
)

# --- Qdrant Client Setup (for Cloud) ---
if not QDRANT_CLOUD_URL:
    QDRANT_CLOUD_URL = getpass.getpass("Enter your Qdrant Cloud Cluster URL (e.g., https://<id>.us-east.aws.cloud.qdrant.io:6333): ")
if not QDRANT_API_KEY:
    QDRANT_API_KEY = getpass.getpass("Enter your Qdrant Cloud API Key: ")

client = QdrantClient(
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_API_KEY,
    # Depending on your Qdrant Cloud setup, you might need to enable prefer_grpc
    # prefer_grpc=True, # Uncomment if you want to prefer gRPC for faster uploads
)


def initialize_qdrant_collection():
    """
    Initializes the Qdrant collection if it doesn't exist.
    """
    dummy_text = "This is a test string to get embedding dimension."
    dummy_embedding = embeddings_model.embed_query(dummy_text)
    vector_size = len(dummy_embedding)

    print(f"Checking for Qdrant collection: '{COLLECTION_NAME}' on {QDRANT_CLOUD_URL}")
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        if collection_info.config.vectors.size != vector_size:
            print(f"Warning: Existing collection '{COLLECTION_NAME}' has vector size {collection_info.config.vectors.size}, but Gemini embeddings are {vector_size}. Recreating collection.")
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            print(f"Collection '{COLLECTION_NAME}' recreated with vector size {vector_size}.")

    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' not found or error accessing: {e}. Creating...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created with vector size {vector_size}.")

def read_all_files(directory_path: str) -> Dict[str, str]:
    """
    Reads all text files from the given directory and returns their content.
    """
    all_docs = {}
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return all_docs

    print(f"Reading files from: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.txt', '.md')):
            print(f"  - Reading: {filename}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_docs[filename] = f.read()
            except Exception as e:
                print(f"    Error reading {filename}: {e}")
    return all_docs

def split_doc_into_chunks(doc_content: str) -> List[str]:
    """
    Splits a document's content into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(doc_content)
    return chunks

def perform_sample_search(query_text: str, top_k: int = 3):
    """
    Performs a sample similarity search in Qdrant using the given query text.
    """
    print(f"\n--- Performing Sample Search Query ---")
    print(f"Search Query: '{query_text}'")

    try:
        # Embed the query text
        # GoogleGenerativeAIEmbeddings.embed_query implicitly uses task_type="retrieval_query"
        query_embedding = embeddings_model.embed_query(query_text)

        # Perform the search in Qdrant
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True # Retrieve the original chunk text and metadata
        )

        if search_results:
            print(f"\nTop {len(search_results)} results:")
            for i, hit in enumerate(search_results):
                print(f"\nResult {i+1} (Score: {hit.score:.4f}):")
                print(f"  Source File: {hit.payload.get('source_file', 'N/A')}")
                print(f"  Chunk Index: {hit.payload.get('chunk_index', 'N/A')}")
                print(f"  Content:")
                print(f"    {hit.payload.get('text_content', 'N/A')[:500]}...") # Print first 500 chars
        else:
            print("No results found for the query.")

    except Exception as e:
        print(f"Error during sample search: {e}")

def process_and_upload_batches():
    """
    Main function to orchestrate reading, splitting, embedding, and uploading.
    """
    initialize_qdrant_collection()
    documents = read_all_files(DATA_DIR)

    total_chunks_processed = 0
    current_batch_texts: List[str] = []
    current_batch_metadata: List[Dict[str, Any]] = []

    for doc_name, doc_content in documents.items():
        print(f"\nProcessing document: {doc_name}")
        chunks = split_doc_into_chunks(doc_content)
        print(f"  - Split into {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            payload = {
                "source_file": doc_name,
                "chunk_index": i,
                "text_content": chunk,
                "chunk_length": len(chunk)
            }
            current_batch_texts.append(chunk)
            current_batch_metadata.append(payload)

            if len(current_batch_texts) >= BATCH_SIZE:
                print(f"    Generating embeddings for batch of {len(current_batch_texts)} chunks...")
                try:
                    batch_embeddings = embeddings_model.embed_documents(current_batch_texts)

                    points_for_this_batch = []
                    for j, embedding in enumerate(batch_embeddings):
                        chunk_text = current_batch_texts[j]
                        payload_data = current_batch_metadata[j]
                        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_name}_{payload_data['chunk_index']}_{chunk_text[:50]}"))

                        points_for_this_batch.append(
                            models.PointStruct(
                                id=chunk_id,
                                vector=embedding,
                                payload=payload_data
                            )
                        )

                    print(f"    Uploading batch of {len(points_for_this_batch)} chunks to Qdrant Cloud...")
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        wait=True,
                        points=points_for_this_batch
                    )
                    total_chunks_processed += len(points_for_this_batch)
                    print(f"    Batch uploaded. Total chunks processed: {total_chunks_processed}")

                    current_batch_texts = []
                    current_batch_metadata = []

                except Exception as e:
                    print(f"    Error processing or uploading batch: {e}")
                    current_batch_texts = []
                    current_batch_metadata = []

    if current_batch_texts:
        print(f"\nGenerating embeddings for final batch of {len(current_batch_texts)} chunks...")
        try:
            final_batch_embeddings = embeddings_model.embed_documents(current_batch_texts)

            points_for_final_batch = []
            for j, embedding in enumerate(final_batch_embeddings):
                chunk_text = current_batch_texts[j]
                payload_data = current_batch_metadata[j]
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_name}_{payload_data['chunk_index']}_{chunk_text[:50]}"))

                points_for_final_batch.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload=payload_data
                    )
                )

            print(f"Uploading final batch of {len(points_for_final_batch)} chunks to Qdrant Cloud...")
            client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=points_for_final_batch
            )
            total_chunks_processed += len(points_for_final_batch)
            print(f"Final batch uploaded. Total chunks processed: {total_chunks_processed}")
        except Exception as e:
            print(f"Error uploading final batch to Qdrant Cloud: {e}")

    print(f"\n--- Script finished ---")
    print(f"Successfully processed and uploaded {total_chunks_processed} chunks to Qdrant collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    #process_and_upload_batches()
    perform_sample_search(query_text="where did harry potter live?")