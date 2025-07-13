from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, AsyncGenerator
import uvicorn
import asyncio
import time
import json
from pydantic import BaseModel
from ingestion import get_hybrid_retriever, prewarm_retrieval_chain
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Retrieval API",
             description="API for hybrid document retrieval using Qdrant")

@app.on_event("startup")
async def startup_event():
    """Pre-warm the retrieval chain on startup"""
    await prewarm_retrieval_chain()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query: str
    k: int = 4

class SearchResult(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.get("/")
async def root():
    return {"message": "Document Retrieval API is running. Use /search endpoint to search documents."}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "CSS Document Retrieval API"
    }

async def generate_stream_response(chain, query: str) -> AsyncGenerator[str, None]:
    """Generate streaming response from the chain"""
    start_time = time.time()
    
    try:
        # Get the response (non-streaming for now)
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chain.invoke({"input": query})
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Yield processing time info
        yield json.dumps({
            "type": "metadata",
            "processing_time": f"{processing_time:.2f} seconds"
        }) + "\n"
        
        # Yield the answer
        if "answer" in response:
            # Split the answer into chunks to simulate streaming
            answer = response["answer"]
            chunk_size = 3  # Characters per chunk
            for i in range(0, len(answer), chunk_size):
                yield json.dumps({
                    "type": "content",
                    "content": answer[i:i+chunk_size]
                }) + "\n"
                await asyncio.sleep(0.001)  # Small delay between chunks
        
        # Add source documents with page information
        sources = []
        for doc in response.get("context", []):
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                content = doc.page_content
                metadata = doc.metadata
                
                # Get page information
                page_number = metadata.get('page', 'N/A')
                page_label = metadata.get('page_label', '')
                source = metadata.get('source', 'Unknown source')
                
                # Format the source information
                source_info = {
                    "page_number": page_number,
                    "page_label": page_label if page_label else f"Page {page_number}",
                    "source": source
                }
                
                # Only add if we have valid content
                if content.strip():
                    sources.append(source_info)
        
        yield json.dumps({
            "type": "sources",
            "sources": sources
        }) + "\n"
        
    except Exception as e:
        logger.error(f"Error in generate_stream_response: {str(e)}")
        yield json.dumps({
            "type": "error",
            "content": f"An error occurred: {str(e)}"
        }) + "\n"

@app.post("/search")
async def search_documents(search_query: SearchQuery):
    try:
        # Get the pre-warmed retrieval chain
        retrieval_data = await prewarm_retrieval_chain(k=search_query.k)
        if not retrieval_data:
            raise HTTPException(status_code=500, detail="Failed to initialize retrieval chain")
            
        retrieval_chain = retrieval_data['retrieval_chain']
        
        # Return the streaming response
        return StreamingResponse(
            generate_stream_response(retrieval_chain, search_query.query),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)