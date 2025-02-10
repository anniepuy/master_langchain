"""
Title: Webpage Loaders
Purpose: Load webpage data through LangChain
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    PromptTemplate
)
from langchain_community.document_loaders import WebBaseLoader
import asyncio
import re

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

clean_context = text_clean(context)
print(clean_context)