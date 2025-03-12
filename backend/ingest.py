import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from vector_db import VectorDB
import config
import sys

class Ingestor:
    def __init__(self):
        """
        Initialize the script ingestor with the directory containing TV show scripts.
        """

        self.db = VectorDB()  # Initialize VectorDB
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


    def process_book_json(self, file_path):
        """
        Reads a book JSON file, splits text into chunks, embeds them, and stores in the vector database.

        Args:
            file_path (str): Path to the JSON file containing book data.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            book_data = json.load(f)

        book_title = book_data["book_title"]
        author = book_data["author"]
        text = book_data["text"]

        # Split text into smaller chunks for better vectorization
        chunks = self.text_splitter.split_text(text)

        # Store each chunk with metadata
        documents = [
            {
                "text": chunk,
                "metadata": {"book_title": book_title, "author": author, "chunk_index": i},
            }
            for i, chunk in enumerate(chunks)
        ]

        # Add to vector database
        self.db.add_documents(documents)
        print(f"Processed and stored embeddings for: {book_title}")

    def ingest_books(self):
        """
        Processes all book JSON files in the books_json directory and adds them to the vector database.
        """
        for filename in os.listdir(config.BOOKS_JSON_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(config.BOOKS_JSON_DIR, filename)
                self.process_book_json(file_path)

if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.ingest_books()