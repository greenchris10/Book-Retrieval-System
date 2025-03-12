import streamlit as st
import matplotlib.pyplot as plt
from backend.topic_modeling import TopicModeling
import pandas as pd
import numpy as np

topic_model = TopicModeling()

st.title("Topic Modeling Visualization")

embeddings, metadata = topic_model.get_embeddings_from_db()

embeddings = np.array(np.array(e) for e in embeddings)

book_titles = [dic['book_title'] for dic in metadata]
texts = [dic["text"] for dic in metadata]
data = {"book_title":book_titles, "texts":texts, "embeddings":embeddings}
df = pd.DataFrame.from_dict(data)

selected_book = st.selectbox("Select a book:", df['book_title'].unique())

if selected_book:
    
    df[df["book_title"] == selected_book]
    
    if not df.empty:
        # Perform topic modeling with BERTopic
        topics, probs = topic_model.bert_model.fit_transform(df['texts'], np.array(df['embeddings'].to_list()))
        
        # Display the topic bar chart
        st.subheader("Topic Distribution Bar Chart")
        fig_barchart = topic_model.bert_model.visualize_barchart()
        st.plotly_chart(fig_barchart)

        # Display the topic overview visualization
        st.subheader("Topic Overview")
        fig_topics = topic_model.bert_model.visualize_topics()
        st.plotly_chart(fig_topics)
    else:
        st.warning("No embeddings found for the selected book. Please ensure embeddings are stored properly.")
