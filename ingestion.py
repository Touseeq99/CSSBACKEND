# Configuration
import os
from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv(override=True)

# Configuration from environment variables
DIRECTORY_PATH = os.getenv('DIRECTORY_PATH', 'documents')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'CSSDOCS')
DENSE_MODEL_NAME = os.getenv('DENSE_MODEL_NAME', 'models/embedding-001')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GOOGLE_API_KEY = SecretStr(os.getenv('GOOGLE_API_KEY', ''))

# Validate required environment variables
required_vars = [
    'QDRANT_URL',
    'QDRANT_API_KEY',
    'GOOGLE_API_KEY',
    'GROQ_API_KEY'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough        # Google embedding‑001 → 768‑D
import os, uuid, asyncio, logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import time
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
    FastEmbedSparse,
)
from pydantic import SecretStr

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# ─── LOGGING ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── ENV ───────────────────────────────────────────────────────────────────
load_dotenv()  # loads GOOGLE_API_KEY if present
executor = ThreadPoolExecutor(max_workers=10)

# ─── LOADERS ───────────────────────────────────────────────────────────────
# Google API key is now loaded from environment variables

def first_64_pages(path: str):
    log.info(f"Loading first 64 pages: {os.path.basename(path)}")
    return PyPDFLoader(path).load_and_split()[:64]


def load_documents(path: str) -> List:
    docs = []
    if os.path.isfile(path):
        if path.lower().endswith(".pdf"):
            docs.extend(first_64_pages(path))
        else:
            docs.extend(DirectoryLoader(path=path, glob="*").load())
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if f.lower().endswith(".pdf"):
                    docs.extend(first_64_pages(fp))
                else:
                    docs.extend(DirectoryLoader(path=fp, glob="*").load())
    else:
        raise FileNotFoundError("DIRECTORY_PATH is not valid")
    log.info(f"Total documents loaded: {len(docs)}")
    return docs


# ─── INGESTION ─────────────────────────────────────────────────────────────
async def ingest() -> Dict[str, Any]:
    if not os.path.exists(DIRECTORY_PATH):
        return {"status": "error", "message": "DIRECTORY_PATH does not exist."}

    def job():
        log.info("🚀 Starting ingestion…")
        documents = load_documents(DIRECTORY_PATH)
        if not documents:
            return {"status": "error", "message": "No documents found."}

        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
        chunks = splitter.split_documents(documents)
        log.info(f"Chunks created: {len(chunks)}")

        # Dense embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=DENSE_MODEL_NAME, google_api_key=GOOGLE_API_KEY
        )

        # Sparse embeddings (BM25)
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Qdrant client & collection
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            log.info(f"Creating collection: {COLLECTION_NAME}")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(size=768, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
        else:
            log.info("Using existing collection")

        # Vector store
        vectordb = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        uuids = [str(uuid.uuid4()) for _ in chunks]
        log.info("Uploading chunks to Qdrant…")
        vectordb.add_documents(documents=chunks, ids=uuids)
        log.info("✅ Ingestion complete.")
        return {"status": "success", "chunks_ingested": len(chunks)}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, job)


# ─── RETRIEVER ─────────────────────────────────────────────────────────────


# Global variable to store the pre-warmed chain
_retrieval_chain_cache = None

async def prewarm_retrieval_chain(k: int = 4):
    """Pre-warm the retrieval chain by making a dummy query"""
    global _retrieval_chain_cache
    if _retrieval_chain_cache is None:
        print("Pre-warming retrieval chain...")
        start_time = time.time()
        chain_data = get_hybrid_retriever(k=k)
        # Make a dummy query to load models
        try:
            await chain_data['retrieval_chain'].ainvoke({"input": "warmup"})
            print(f"Retrieval chain pre-warmed in {time.time() - start_time:.2f} seconds")
            _retrieval_chain_cache = chain_data
        except Exception as e:
            print(f"Error during pre-warming: {e}")
            _retrieval_chain_cache = chain_data  # Still cache even if warmup fails
    return _retrieval_chain_cache

def get_hybrid_retriever(k: int = 4, alpha: float = 0.5):
    """
    Create a hybrid retriever with ChatGroq LLM for generating responses.
    
    Args:
        k: Number of documents to retrieve
        alpha: Weight for hybrid search (0 = dense only, 1 = sparse only)
    
    Returns:
        A dictionary with 'retriever' and 'qa_chain' for question answering
    """
    log.info("Initializing hybrid retriever with ChatGroq LLM...")
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=DENSE_MODEL_NAME, google_api_key=GOOGLE_API_KEY
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # Initialize Qdrant vector store
    store = QdrantVectorStore(
        client=QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY),
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    
    # Create retriever
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "lambda_mult": alpha  # Controls balance between relevance and diversity
        },
    )
    
    # Initialize ChatGroq LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3,  # Slightly higher temperature for more varied responses
        max_tokens=2048,
        api_key=GROQ_API_KEY
    )
    
    # Enhanced prompt for Pakistan Civil Services tutor
    prompt_template = """You are an expert tutor for Pakistan Civil Services (CSS) aspirants. 
    Your role is to provide clear, well-structured, and comprehensive answers to help students prepare for their exams.
    
    Context from study materials:
    {context}
    
    Student's Question: {input}
    
    Guidelines for your response:
    1. Begin with a brief, clear answer to the question
    2. Provide detailed explanation with relevant examples
    3. Include important dates, names, and facts where applicable
    4. Structure your response with clear paragraphs
    5. If the question is about Pakistan, focus on the Pakistani context
    6. If you don't know the answer, be honest and suggest where to find the information
    7. End with a summary or key takeaway
    
    CSS Tutor's Response:"""
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "input"]
    )
    
    # Create document processing chain
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    
    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    
    return {
        'retriever': retriever,
        'retrieval_chain': retrieval_chain
    }


# ─── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = asyncio.run(ingest())
    log.info(f"\nResult: {result}")
