# CSS Document Retrieval Backend

A FastAPI-based backend for document retrieval and question-answering using Qdrant vector store and ChatGroq LLM.

## Features

- Hybrid document retrieval using both dense and sparse embeddings
- Streaming responses for better user experience
- Pre-warmed retrieval chain for faster responses
- Environment-based configuration

## Prerequisites

- Python 3.9+
- Qdrant Cloud account (or self-hosted Qdrant)
- Groq API key
- Google API key for embeddings

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Touseeq99/CSSBACKEND.git
   cd CSSBACKEND
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and configuration
   ```bash
   copy .env.example .env  # On Windows
   # or
   cp .env.example .env   # On Linux/Mac
   ```

## Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Health check
- `POST /search`: Search documents and get answers
  ```json
  {
    "query": "Your question here",
    "k": 4
  }
  ```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| QDRANT_URL | Qdrant server URL | Yes | - |
| QDRANT_API_KEY | Qdrant API key | Yes | - |
| COLLECTION_NAME | Qdrant collection name | No | CSSDOCS |
| DIRECTORY_PATH | Path to documents | No | documents |
| DENSE_MODEL_NAME | Embedding model | No | models/embedding-001 |
| GROQ_API_KEY | Groq API key | Yes | - |
| GOOGLE_API_KEY | Google API key for embeddings | Yes | - |

## Security

- Never commit your `.env` file
- Add `.env` to your `.gitignore`
- Use environment variables for sensitive information

## License

MIT