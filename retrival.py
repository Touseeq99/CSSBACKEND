import logging
import time
from ingestion import (
    get_hybrid_retriever,          # we reuse the helper you already wrote
    QDRANT_URL, QDRANT_API_KEY,GOOGLE_API_KEY    # reuse constants for consistency
)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info("Starting hybrid retrieval example...")
    query = "How does Plato's critique of Athenian democracy reflect the broader tension in political thought between idealism and realism, as introduced in the opening chapter?"
    start_time = time.time()
    log.info(f"\n🔎 Query: {query}")
    retriever = get_hybrid_retriever(k=3, alpha=0.5)

    
    docs = retriever.invoke(query)
    
    for i, d in enumerate(docs, 1):
        print(f"\n--- Answer {i} ---\n{d.page_content[:500]}…")
    
    elapsed_time = time.time() - start_time
    log.info(f"Retrieval completed in {elapsed_time:.2f} seconds.")



