# Book-Retrieval-System

Overview:

This project is an LLM-powered retrieval augmented generation system that enables natural language querying of Frank Herbert's Dune book series. Using LangChain, Milvus (Zilliz), and Ollama, the system retrieves relevant text embeddings and generates context-aware responses using open-source LLMs (LLaMA 3.2). It's still a work in progress so some of the files may not make the most sense at the moment. I'm hoping to add some visualization and analysis with BertTopic.

Key Features:

  LLM-Powered Responses: Uses LangChain & Ollama to generate intelligent, contextual responses.
  
  Scalable Vector Search: Stores high-dimensional text embeddings in Milvus (Zilliz) vector database for fast & efficient retrieval.
  
  Built with Streamlit, featuring an interactive search and visualization interface.
  
  Fully Dockerized for easy setup and scalability.

Guide:

1️⃣ Install Dependencies

pip install -r requirements.txt

2️⃣ Set Up Environment Variables

Create a .env file for the paths listed in the config.py file.

3️⃣ Run the Dockerized Application

docker-compose up --build

4️⃣ Access the Web App

Open your browser and go to:

http://localhost:8501

