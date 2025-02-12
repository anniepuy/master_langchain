"""
Title: Using LangChain Search Tools
Purpose: Tool calling to support agents
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
Requires pip install duckduckgo-search wikipedia yfinance xmltodict tavily-python

"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#base set up
base_url = "http://localhost:11434"

model = 'llama3.2'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 1,
)

## Using DuckDuckGo Search 
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

result = search.invoke("What is today's stock market news?")

print(f"Duck duck go: {result}")

##Using Taviley
from langchain_community.tools import TavilySearchResults

tavily_search = TavilySearchResults(
    max_results =5,
    search_depth = "advanced",
    include_answer=True,
    include_raw_content=True,
)

tavily_question = "What is today's stock market news?"

tavily_result = tavily_search.invoke(tavily_question)
print('\n\n')
print(f"Tavily: {tavily_result}")

### Using Wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wiki_question = "What is today's stock market news?"

wiki_result = wiki_search.invoke(wiki_question)
print('\n\n')
print(f"Wiki: {wiki_result}")

### Using PubMed
from langchain_community.tools import PubmedQueryRun
pubmed_search = PubmedQueryRun()

pubmed_question = "What is latest research on COVID-19?"

print('\n\n')
print(f"PubMed: {pubmed_search.invoke(pubmed_question)}")

