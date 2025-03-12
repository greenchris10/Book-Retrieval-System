import pandas as pd
import numpy as np
import umap
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from backend.vector_db import VectorDB  # Import Zilliz database handler
from bertopic import BERTopic

class TopicModeling:
    def __init__(self):
        self.vector_db = VectorDB()  # Connect to Zilliz
        self.bert_model = BERTopic()

    def get_embeddings_from_db(self):
        """Retrieve stored embeddings and metadata from the Zilliz vector database."""
        return self.vector_db.get_all_embeddings()

    def fit_transform(self, texts, embeddings):
        """
        Fit BERTopic model on precomputed embeddings.
        """
        embeddings = np.array(embeddings)  # Ensure embeddings are in NumPy format
        topics, probs = self.bert_model.fit_transform(texts, embeddings)
        return topics, probs