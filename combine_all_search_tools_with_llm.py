"""
Title: Combining all tools and binding with LLM
Purpose: Tool calling to support agents
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
Requires pip install duckduckgo-search wikipedia yfinance xmltodict tavily-python

"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import PubmedQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool

load_dotenv()

#base set up
base_url = "http://localhost:11434"

model = 'llama3.2'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 1,
)

### Creating multiple tools
@tool
def wikipedia_search(query):
    """
    Search Wikipedia for general information.

    Args:
    query: The search query
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    response = wikipedia.invoke(query)
    return response

@tool
def pubmed_search(query):
    """
    Search PubMed for medical and life sciences queries.

    Args:
    query: The search query
    """
    search = PubmedQueryRun()
    response = search.invoke(query)
    return response

@tool
def tavi_search(query):
    """
    Search the web for realtime and latest information. For example, stock market news, news, weather, etc.
    
    Args:
    query: The search query
    """
    search = TavilySearchResults(
        max_results =5,
        search_depth = "advanced",
        include_answer=True,
        include_raw_content=True,
    )

    response = search.invoke(query)
    return response

#Our custom tool
@tool
def multiply(a:int, b:int)-> int:
    """
    Multiply two integers numbers together

    Args:
    a: First integer
    b: Second integer
    """
    return int(a) * int(b)

#Create a list of tools
tools = [wikipedia_search, pubmed_search, tavi_search, multiply]

#Bind the tools to the LLM
list_of_tools = {tool.name: tool for tool in tools}

#bind to LLM
llm_with_tools = llm.bind_tools(tools)

##now call the tools with various queries
#query = "what is the latest news?"
#query = "what is todays stock market news?"
#query = "what is the latest research on lung cancer?"
#query = "what is the product of 2 and 3?"
query = "What is an LLM?"

response = llm_with_tools.invoke(query)
#prints which tool is used
#print(response)
#print(response.tool_calls)

## Pass the tool result to the LLM - Final Output
from langchain_core.messages import HumanMessage

#place queries inside human message.
messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)

messages.append(ai_msg)
#print(messages)

## With multiple questions you must iterate over the tools
for tool_call in ai_msg.tool_calls:
    print(tool_call)

    name = tool_call['name'].lower()
    selected_tool = list_of_tools[name]
    tool_msg = selected_tool.invoke(tool_call['args'])
    messages.append(tool_msg)

response = llm_with_tools.invoke(messages)
print(response.content)