from langchain_huggingface import HuggingFaceEmbeddings
from backend.vector_db import VectorDB
import backend.config as config
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

class SemanticSearch:
    def __init__(self, model_name=config.EMBEDDING_MODEL, top_k=5):
        
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_db = VectorDB()
        self.top_k = top_k
        self.prompt = self.get_prompt()
        self.llm = Ollama(model="llama3.2:1b", temperature=0.2)
        self.chain = self.get_chain()

    def get_prompt(self):
        prompt = PromptTemplate(input_variables=["context", "question"],
                                template=(
                                            "You are an AI assistant knowledgeable about the *Dune* books. "
                                            "Use the provided context to answer the question accurately. "
                                            "If the answer isn't found in the context, say so.\n\n"
                                            "Context:\n{context}\n\n"
                                            "Question: {question}\nAnswer:"
                                            ),)
        return prompt
    
    def get_chain(self):
        chain = (
            RunnableLambda(self.get_query)
            | RunnablePassthrough.assign(context=lambda x: x["context"], question=lambda x: x["question"])
            | self.prompt
            | self.llm  
        )
        return chain
    
    def get_query(self, query):

        results = self.vector_db.search(query['question'])

        return {"question": query, "context": "/n".join(results)}
    
    def answer_query(self, query):
        return self.chain.invoke({"question": query})
    
search_engine = SemanticSearch()