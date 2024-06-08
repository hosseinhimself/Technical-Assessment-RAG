from fastapi import FastAPI, HTTPException
import os
from RAGModel import DocumentProcessor, EmbeddingModel, FaissIndex, FaissRetriever, QueryEngine, Document

# FastAPI app setup
app = FastAPI(
    title="Technical Assessment for RAG API",
    description="An API for the RAG model with document addition and query capabilities.",
    version="1.0.0",
    docs_url="/docs",  
    redoc_url="/redoc"
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
    Retrieves context related to the query text.
    - **query_text**: The text to query for relevant documents.
    - **top_k**: The number of top documents to retrieve.
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
    """
    try:
        query_engine.add_document(document.text)
        return {"message": "Document added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
