from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, Collection
import backend.config as config
from typing import List, Dict
import sys, os
import numpy as np

class VectorDB:
    def __init__(self, collection_name: str = "book_collection"):
       
        self.collection_name = collection_name
        self._connect_milvus()
        self._initialize_vector_store()
        self.collection = Collection(collection_name)

    def _connect_milvus(self):

        connections.connect(
            uri = config.ZILLIZ_CLOUD_URI,
            token=config.ZILLIZ_CLOUD_API_KEY
        )

    def _initialize_vector_store(self):
        
        self.vector_store = Milvus(
            embedding_function=HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL),
            collection_name=self.collection_name,
            connection_args={"uri": config.ZILLIZ_CLOUD_URI, "token": config.ZILLIZ_CLOUD_API_KEY},
            auto_id=True
        )

    def add_documents(self, documents):
        """Adds documents (chunks) to the vector database."""
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        self.vector_store.add_texts(texts, metadatas=metadatas)
        print(f"Stored {len(documents)} chunks in vector database.")

    def insert_embeddings(self, texts: List[str], metadata: List[Dict]):
        """
        Inserts text and corresponding metadata into Milvus.

        Args:
            texts (List[str]): List of script dialogues or scene descriptions.
            metadata (List[Dict]): List of metadata dictionaries (season, episode, character, etc.).
        """
        self.vector_store.add_texts(texts=texts, metadatas=metadata)
        print(f"Inserted {len(texts)} embeddings into Milvus.")

    def search(self, query: str, top_k: int = 5, filters: Dict = None):
        """
        Performs semantic search on stored embeddings.

        Args:
            query (str): The user's search query.
            top_k (int): Number of results to return.
            filters (Dict): Optional metadata filters (e.g., {"book_title": "Dune: Messiah", "author": "Frank Herbert"}).

        Returns:
            List[Dict]: Top-k matching texts with metadata.
        """
        results = self.vector_store.similarity_search(query, k=top_k, filter=filters)
        
        output = []
        for result in results:
            output.append(result.page_content)
        return output
    
    def get_all_embeddings(self):
        """Retrieve all stored embeddings and metadata from the database.
        
        Returns:
            List[Dict]: List of dictionaries containing booktitle and text
        """
        self.collection.load()

        results = self.collection.query(expr="", output_fields=["vector", "book_title", "text"], limit = 300)

        if not results:
            return None, None

        embeddings = [res["vector"] for res in results]
        metadata = [{"book_title": res["book_title"], "text": res["text"]} for res in results]

        return embeddings, metadata
    

if __name__ == "__main__":
    db = VectorDB()
    print("Milvus connection successful.")
