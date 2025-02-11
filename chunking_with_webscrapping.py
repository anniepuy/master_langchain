"""
Title: Chunking with Webscrapping
Purpose: Applying chunking for better LLM performance
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""


from dotenv import load_dotenv
from langchain_core.prompts import (
    PromptTemplate
)
from langchain_community.document_loaders import WebBaseLoader
import asyncio
import re
from scripts import llm_qa_script

load_dotenv()

#list of URLS
urls = [
    'https://www.barrons.com/livecoverage/stock-market-today-021025',
    'https://economictimes.indiatimes.com/markets/stocks/news',
    'https://www.marketwatch.com/livecoverage/stock-market-today-dow-s-p-500-and-nasdaq-set-for-partial-recovery-from-friday-s-losses?mod=home_editorspick',
]

loader = WebBaseLoader(web_path=urls)

#read the data

async def load_documents():
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs

# Run the async function in a blocking manner
docs = asyncio.run(load_documents())

def format_docs(docs):
    return "\n\n".join([x.page_content for x in docs])

#whole context
context = format_docs(docs)
#print(context)

#format the output - remove new lines.
def text_clean(text):
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\t+', '\t', text)
    text = re.sub(r'\s+', ' ', text)
    return text

context = text_clean(context)


## using the imported LLM script
#response = llm_qa_script.ask_llm(context, question = "What is today's news?")

#print(response)

#changing the output through chunking
response = llm_qa_script.ask_llm(context[:10_000], question = "What is today's news?")

print(response)

#making a function for chunking with overlap
def chunk_text(text, chunk_size, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        yield text[i:i + chunk_size]
    return chunks

chunks = chunk_text(context, 10_000)

#print(chunks)

##Now we can pass indvidual chunks to the LLM model

#Question
question = "What is today's top stock market news?"
chunk_summary = []
for chunk in chunks:
    response = llm_qa_script.ask_llm(chunk, question)
    chunk_summary.append(response)

#print(chunk_summary)

summary = "\n\n".join(chunk_summary)

#print(summary)

#Combine reports for final report
question2 = "Write a detailed news report from the given context."

response = llm_qa_script.ask_llm(summary, question2)

import os
os.makedirs("reports", exist_ok=True)
with open("reports/news_report.md", "w") as f:
    f.write(response)
