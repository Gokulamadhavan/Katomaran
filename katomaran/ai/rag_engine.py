from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class RAGEngine:
    def __init__(self):
 
        self.embeddings = OpenAIEmbeddings()

        self.vectorstore = FAISS.from_texts([], self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=self.vectorstore.as_retriever())
    
    def answer_query(self, query: str) -> str:
        return self.qa_chain.run(query)
