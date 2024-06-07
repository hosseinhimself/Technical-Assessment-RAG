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

class Document(BaseModel):
    text: str

class DocumentProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.documents = self.load_documents()

    def load_documents(self):
        documents = SimpleDirectoryReader(self.directory).load_data()
        return self.refine_documents(documents)

    def refine_documents(self, documents):
        refined_docs = []
        for doc in documents:
            if "Member-only story" in doc.text or "The Data Entrepreneurs" in doc.text or " min read" in doc.text:
                continue
            refined_docs.append(doc)
        return refined_docs

    def get_texts(self):
        return [doc.text for doc in self.documents]

    def add_document(self, document):
        refined_doc = self.refine_documents([document])
        if refined_doc:
            self.documents.extend(refined_doc)
            return refined_doc[0]
        return None

class EmbeddingModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings

class FaissIndex:
    def __init__(self, dimension=None): 
        self.dimension = dimension
        self.index = None

    def add_embeddings(self, embeddings):
        if self.dimension is None:
            self.dimension = embeddings.shape[1]  
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def load(self, index_file, documents_file, embeddings_file):
        # Existing code
        self.index = faiss.read_index(index_file)
        with open(documents_file, "rb") as f:
            documents = pickle.load(f)
        embeddings = np.load(embeddings_file)
        self.dimension = embeddings.shape[1] 
        return documents, embeddings

    def add_document(self, document, embedding):
        self.add_embeddings(np.array([embedding]))

class QueryEngine:
    def __init__(self, retriever, embedding_model):
        self.retriever = retriever
        self.embedding_model = embedding_model

    def query(self, query_text):
        query_embedding = self.embedding_model.get_embeddings([query_text])[0]
        retrieved_documents = self.retriever.retrieve(query_embedding)
        context = ""
        for doc in retrieved_documents:
            context += doc.text + "\n\n"
        return context

    def add_document(self, document_text):
        # Create a new Document object
        new_document = Document(text=document_text)
        # Add the document to the DocumentProcessor
        refined_document = self.retriever.documents.append(new_document)
        if refined_document:
            # Generate the embedding for the new document
            new_embedding = self.embedding_model.get_embeddings([document_text])[0]
            # Add the document and its embedding to the FAISS index
            self.retriever.index.add_document(refined_document, new_embedding)

class FaissRetriever:
    def __init__(self, index, documents, top_k):
        self.index = index
        self.documents = documents
        self.top_k = top_k

    def retrieve(self, query_embedding):
        distances, indices = self.index.search(np.array([query_embedding]), self.top_k)
        return [self.documents[idx] for idx in indices[0]]