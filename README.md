# Technical-Assessment-RAG

This document retrieval system efficiently retrieves relevant documents based on user queries. It leverages the RAG (Retrieval-Augmented Generation or Retrieve-and-Generate) model architecture, utilizing transformers for text embeddings and FAISS for efficient similarity search.

### Components

**RAGModel.py**: This module contains classes and functions related to document processing, embedding generation, indexing, and query retrieval.

- **Document**: A Pydantic model representing a document with a single attribute text.
- **DocumentProcessor**: Loads and refines documents from a given directory.
- **EmbeddingModel**: Handles the loading of a transformer-based model for generating embeddings.
- **FaissIndex**: Manages the FAISS index for efficient document retrieval based on embeddings.
- **QueryEngine**: Performs document retrieval based on a query using the embedding model and Faiss index.
- **FaissRetriever**: Retrieves documents from the Faiss index based on query embeddings.

**main.py**: This file sets up a FastAPI app for serving queries and adding documents.

- Initializes necessary components like the document processor, embedding model, Faiss index, and query engine.
- Defines routes for querying and adding documents.
- Loads or creates the Faiss index and related files if they don't exist.
- Initializes the retriever with the Faiss index and documents.
- Initializes the query engine with the retriever and embedding model.

**test.py**: A script to test the query endpoint of the FastAPI app by sending a query and printing the response.

- Sends a GET request to the /query endpoint with a query text.
- Prints the response received from the server.

### Usage

1. **Installation**: Install the required dependencies by running:

    ```
    pip install -r requirements.txt
    ```

2. **Run the Server**: Execute main.py to run the FastAPI server:

    ```
    python main.py
    ```

3. **Send Queries**: Send queries to the server using test.py or any HTTP client.

### Endpoints

- **/query**: Accepts GET requests with a query text and returns relevant documents.
- **/add_document**: Accepts POST requests with a document text to add new documents to the system.

### Example

```python
import requests

url = "http://localhost:8000/query"
query_text = 'What is fat-tailedness?'

response = requests.get(url, params={"query_text": query_text, "top_k": 5})

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
```

### Configuration

- The directory containing documents is specified in main.py.
- The transformer model used for embeddings is specified in main.py as well.
- The number of top-k documents retrieved per query can be adjusted in main.py.
- Faiss index and related files' paths are defined in main.py.