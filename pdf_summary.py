"""
Title: PDF Summary
Purpose: Make a different prompt for summarization outside of Q&A
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

##Question Answering using LLM
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
import os
import tiktoken

load_dotenv()

#base set up
base_url = "http://localhost:11434"

model = 'llama3.2'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 1,
    num_predict = 256,
)
#load all documents from a directory one by one
#read the list of PDFs in the directory, iterate over the files. Works with nested folder structure as well
pdfs = []
for root, dirs, files in os.walk("./rag_documents_health"):
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root, file))

print(len(pdfs))

#read the documents create an empty list, then iterate through the files and read into the docs 
docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    temp = loader.load()
    docs.extend(temp)

#combine all the PDFS into one context docs, iterate over the page content key
def format_text(docs):
    return ("\n\n".join([x.page_content for x in docs]))

context = format_text(docs)


#To find how many tokens are in the document - use tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

#Prepare system prompt and prompt template for QA
system = SystemMessagePromptTemplate.from_template("""
                            You are a helpful AI assistant who works as a document summarizer.
                            You must not hallucinate or provide any false information.
                            """)    

prompt = """ Summarize the given context in {words}.
        ### Context: 
        {context}
        
        ### Summary:"""

prompt = HumanMessagePromptTemplate.from_template(prompt)

messages = [system, prompt]
template = ChatPromptTemplate(messages)

print(template)

summary_chain = template | llm | StrOutputParser()

response = summary_chain.invoke({'context': context, 'words': 50})

print(response)