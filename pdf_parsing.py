"""
Title: PDF Parsing with PyMuPDF
Purpose: Parsing with PyMU PDF
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
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

#load a single document
#loader = PyMuPDFLoader("./rag_documents_health/SuddenCardiacDeathofAthletesandPre-ParticipationScreeningTheYouthLeagueCoachPerspective.pdf")

#docs = loader.load()

#print(len(docs))

#metadata of the document
#print(docs[0].metadata)

#prints first page of the document
#print(docs[0].page_content)

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

#example - apply the encodings to the text
print(encoding.encode("Hello, World!"))

#how many tokens in the first doc
print(len(encoding.encode(docs[0].page_content)))

#how many tokens in our context
print(len(encoding.encode(context)))