from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

import faiss
import numpy as np
import torch
import pickle

# Define a Pydantic model for the document
class Document(BaseModel):
    text: str

# Class to process documents from a directory
class DocumentProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.documents = self.load_documents()

    # Load and refine documents from the directory
    def load_documents(self):
        documents = SimpleDirectoryReader(self.directory).load_data()
        return self.refine_documents(documents)

    # Refine documents by filtering out unwanted content
    def refine_documents(self, documents):
        refined_docs = []
        for doc in documents:
            if "Member-only story" in doc.text or "The Data Entrepreneurs" in doc.text or " min read" in doc.text:
                continue
            refined_docs.append(doc)
        return refined_docs

    # Get the text content of the documents
    def get_texts(self):
        return [doc.text for doc in self.documents]

    # Add a new document after refining it
    def add_document(self, document):
        refined_doc = self.refine_documents([document])
        if refined_doc:
            self.documents.extend(refined_doc)
            return refined_doc[0]
        return None

# Class to generate embeddings using a pre-trained model
class EmbeddingModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    # Generate embeddings for the given texts
    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings

# Class to handle FAISS index operations
class FaissIndex:
    def __init__(self, dimension=None): 
        self.dimension = dimension
        self.index = None

    # Add embeddings to the FAISS index
    def add_embeddings(self, embeddings):
        if self.dimension is None:
            self.dimension = embeddings.shape[1]  
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    # Save the FAISS index, documents, and embeddings to disk
    def save(self, index_file, documents_file, embeddings_file, documents):
        faiss.write_index(self.index, index_file)
        with open(documents_file, "wb") as f:
            pickle.dump(documents, f)
        np.save(embeddings_file, self.index.reconstruct_n(0, self.index.ntotal))

    # Load the FAISS index, documents, and embeddings from disk
    def load(self, index_file, documents_file, embeddings_file):
        self.index = faiss.read_index(index_file)
        with open(documents_file, "rb") as f:
            documents = pickle.load(f)
        embeddings = np.load(embeddings_file)
        self.dimension = embeddings.shape[1]
        return documents, embeddings

    # Add a single document embedding to the FAISS index
    def add_document(self, document, embedding):
        self.add_embeddings(np.array([embedding]))

# Class to handle the querying and adding documents to the index
class QueryEngine:
    def __init__(self, retriever, embedding_model, faiss_index, index_file, documents_file, embeddings_file):
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.index_file = index_file
        self.documents_file = documents_file
        self.embeddings_file = embeddings_file

    # Query the index with a text and retrieve relevant documents
    def query(self, query_text):
        query_embedding = self.embedding_model.get_embeddings([query_text])[0]
        retrieved_documents = self.retriever.retrieve(query_embedding)
        context = ""
        for doc in retrieved_documents:
            context += doc.text + "\n\n"
        return context

    # Add a new document to the index
    def add_document(self, document_text):
        new_document = Document(text=document_text)
        refined_document = self.retriever.documents.append(new_document)
        if refined_document:
            new_embedding = self.embedding_model.get_embeddings([document_text])[0]
            self.faiss_index.add_document(new_embedding)
            self.faiss_index.save(self.index_file, self.documents_file, self.embeddings_file, self.retriever.documents)

# Class to handle retrieval of documents using FAISS
class FaissRetriever:
    def __init__(self, index, documents, top_k):
        self.index = index
        self.documents = documents
        self.top_k = top_k

    # Retrieve top_k documents based on the query embedding
    def retrieve(self, query_embedding):
        distances, indices = self.index.search(np.array([query_embedding]), self.top_k)
        return [self.documents[idx] for idx in indices[0]]