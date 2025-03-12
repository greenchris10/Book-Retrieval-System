import streamlit as st
import asyncio
from frontend import search_ui


# Set page title
st.set_page_config(page_title="Dune Book Search", layout="wide")

# Sidebar Navigation
st.sidebar.title("Dune Book Search")
page = st.sidebar.radio("Go to", ["Search"])

if page == "Search":
    search_ui.render_search_page()

