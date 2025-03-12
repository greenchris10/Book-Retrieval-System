import streamlit as st
from backend.search import SemanticSearch


def render_search_page():
    st.title("📖 Dune AI Assistant")
    st.write("Search and analyze passages in the Dune books using natural language.")

    query = st.text_input("Ask a question about the Dune books:", "")

    if st.button("Submit"):
        if query:
            sem = SemanticSearch()
            with st.spinner("Generating response..."):
                response = sem.answer_query(query)
                st.write("### AI Response:")
                st.write(response)
        else:
            st.warning("Please enter a question.")