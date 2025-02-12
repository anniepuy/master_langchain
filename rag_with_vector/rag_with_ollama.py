"""
Title: Rag with Ollama 
Purpose: takes teh RAG/d documents set up in embedding_faiss_setup_standalone.py and uses the LLM to answer questions
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""


from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

import os
import warnings
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

##Hub will pull the prompts from: https://smith.langchain.com/hub/daethyra/rag-prompt
#from langchain import hub

prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use 5 sentences maximum and keep the answer concise. Use bullett points.
        Question: {question} 
        Context: {context} 
        Answer:"""

prompt = ChatPromptTemplate.from_template(prompt)

#get llm 
#base set up
base_url = "http://localhost:11434"

model = 'llama3.2'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 1
)

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
db_name_path = "/Users/annhagan/Source/master_langchain/health_supplements"
vector_store = FAISS.load_local(db_name_path, embeddings, allow_dangerous_deserialization=True)

### Simularity Retreival
question = "how do I gain muscle mass?"
docs = vector_store.search(question, k=5, search_type="similarity")

### convert vector store to retrieval for LangChain Runnables
retriever = vector_store.as_retriever(search_type="similarity", 
                                      search_kwargs = {'k': 5})


#combine the entire document into a single text context to pass the q/a prompt
def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])
context = format_docs(docs)
#print(context)

##Create the runnable chain to pass the chunks and questions to the VectorStore and LLM
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#give it a quesiton and then run the chain with the question
question = "how to lose weight?"
response = rag_chain.invoke(question)

print(response)

question2 = "what supplements are best for atheletes?"
response2 = rag_chain.invoke(question2)
print(response2)