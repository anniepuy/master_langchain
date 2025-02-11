"""
Title: LangChain Job Description RAG - DOCX
Purpose: Learn how to load Microsoft document and create a personalized email
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from langchain_community.document_loaders import Docx2txtLoader
from scripts import llm

loader = Docx2txtLoader("ms_data/job_description.docx")

docs = loader.load()
print(len(docs))

context = docs[0].page_content
print(context)

#now ask a question
question = "My name is Carla and I am recent graduate from univeristy with focus on AI. I am applying for a Data Science positoin at SpiceJet. Please write a concise job application email for me in short, removing any placeholders, and including references to a job board or sources."
response = llm.ask_llm(context, question)
print(response)