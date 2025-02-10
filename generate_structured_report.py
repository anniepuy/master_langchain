"""
Title: Generate Structured Report
Purpose: Generate a structured report from a set of PDFs
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

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
    temperature = 0.8,
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
                            You are a helpful AI assistant who answers user question based on provided content.
                            Do not answer in more than {words} words.
                            """)    

prompt = """ Answer user question based on the provided context only.  If you do not know the answer, just say "I don't know".
        ### Context: 
        {context}

        ### Question: 
        {question}

        ### Answer:"""

prompt = HumanMessagePromptTemplate.from_template(prompt)

messages = [system, prompt]
template = ChatPromptTemplate(messages)

print(template)
template.invoke({'context': context, 'question': 'How to gain muscle mass?', 'words': 50})

qna_chain = template | llm | StrOutputParser()

response = qna_chain.invoke({'context': context, 
                             'question': 'Provide a detailed report from the provided context. Write answer in Markdown.', 
                             'words': 2000})

print(response)