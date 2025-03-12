from sentence_transformers import SentenceTransformer
import torch
from typing import List

class BookEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initializes the embedding model.

        Args:
            model_name (str): Name of the SentenceTransformer model.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Converts a single text input into an embedding.

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: The generated embedding vector.
        """
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a batch of texts into embeddings.

        Args:
            texts (List[str]): List of texts to be embedded.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return self.model.encode(texts, convert_to_numpy=True).tolist()

if __name__ == "__main__":
    embedder = BookEmbedder()
    sample_text = "This is a test book dialogue."
    embedding = embedder.generate_embedding(sample_text)
    print(f"Sample Embedding: {embedding[:5]} ...")  # Print first 5 values for brevity