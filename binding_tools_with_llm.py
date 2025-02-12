"""
Title: Binding tools to be called by LLM
Purpose: Tool calling to support agents
Author: Ann Hagan - via learning through Laxmi Kant on Udemy

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

## Creat our own tools
# This import allows all python code to be a tool
from langchain_core.tools import tool

# tool must be understandble with a description & args, the @tool makes it a runnable

@tool
def add(a, b):
    """
    Add two integers numbers together

    Args: 
    a: First integer
    b: second integer
    """
    return a + b
@tool
def multiply(a, b):
    """
    Multiply two integers numbers together

    Args:
    a: First integer
    b: Second integer
    """
    return a * b

#checking out our tool
#print(add.name, add.description, add.args, add.args_schema.schema())

## To call the tool, we need to add a dictionary
#result = add.invoke({'a': 2, 'b': 3})

#print(result)

## 1. Bind the tools to the LLM so the LLM can call and use the tool
tools = [add, multiply]

llm_with_tools = llm.bind_tools(tools)

#print(llm_with_tools)

### 2. Next ask a question using the tool
question = "What is the sum of 2 and 3?"
question2 = "What is the product of 2 and 3, also what is 11 plus 33?"

#Returns which tool it is using (no content to the question)
print(llm_with_tools.invoke(question).tool_calls)
print(llm_with_tools.invoke(question2).tool_calls)

