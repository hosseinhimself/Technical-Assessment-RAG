from fastapi import FastAPI, HTTPException
import os
from RAGModel import DocumentProcessor, EmbeddingModel, FaissIndex, FaissRetriever, QueryEngine, Document


# Description in plain text format
description = """

This API provides functionality for a Retrieval-Augmented Generation (RAG) model. The main features include querying a corpus for relevant documents and adding new documents to the index.

## Features

- Query Documents: Retrieve the most relevant documents based on a query.
- Add Documents: Add new documents to the index to enhance the corpus.

## Endpoints

- /query: Retrieve relevant documents based on a query text.
- /add_document: Add a new document to the index.

## Example Usage

### Querying Documents

To query documents, send a GET request to the `/query` endpoint with the query text and the desired number of top documents to retrieve:

```
curl -X GET "http://localhost:8000/query?query_text=What+is+fat-tailedness?&top_k=5" -H "accept: application/json"
```

### Adding Documents

To add a new document, send a POST request to the `/add_document` endpoint with the document text in JSON format:

```
curl -X POST "http://localhost:8000/add_document" -H "Content-Type: application/json" -d '{"text": "This is a new document to add."}'
```
"""

# FastAPI app setup
app = FastAPI(
    title="RAG API",
    description=description,
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Directory setup for saving models and related files
model_dir = "assets"
os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define paths for index, documents, and embeddings
index_file = os.path.join(model_dir, "faiss_index.index")
documents_file = os.path.join(model_dir, "documents.pkl")
embeddings_file = os.path.join(model_dir, "embeddings.npy")

# Initialize components
directory = "content"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
default_top_k = 3

# Conditional loading/creation of FAISS index and documents
if not (os.path.exists(index_file) and os.path.exists(documents_file) and os.path.exists(embeddings_file)):
    # Process documents
    doc_processor = DocumentProcessor(directory)
    texts = doc_processor.get_texts()
    print(f"Number of documents after refinement: {len(texts)}")

    # Generate embeddings
    embedding_model = EmbeddingModel(model_name)
    embeddings = embedding_model.get_embeddings(texts)

    # Initialize FAISS index and add embeddings
    faiss_index = FaissIndex(dimension=embeddings.shape[1])
    faiss_index.add_embeddings(embeddings)

    # Save index, documents, and embeddings
    faiss_index.save(index_file, documents_file, embeddings_file, doc_processor.documents)
else:
    faiss_index = FaissIndex() 
    embedding_model = EmbeddingModel(model_name)

# Load index, documents, and embeddings
documents, embedding = faiss_index.load(index_file, documents_file, embeddings_file)

# Initialize retriever
retriever = FaissRetriever(faiss_index.index, documents, default_top_k)
   
# Initialize query engine
query_engine = QueryEngine(retriever, embedding_model, faiss_index, index_file, documents_file, embeddings_file)



@app.get("/query")
async def query(query_text: str, top_k: int):
    """
    Retrieve context related to the query text.
    
    - **query_text**: The text to query for relevant documents.
    - **top_k**: The number of top documents to retrieve (default is 3).
    
    **Returns**: 
    - **context**: The most relevant documents related to the query text.
    """
    try:
        retriever.top_k = top_k
        context = query_engine.query(query_text)
        return {"context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/add_document")
async def add_document(document: Document):
    """
    Adds a new document to the index.
    
    - **document**: The document to be added.
    
    **Returns**: 
    - **message**: Confirmation message indicating the document was added successfully.
    """
    try:
        query_engine.add_document(document.text)
        return {"message": "Document added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
