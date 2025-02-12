"""
Title: Create custom tools
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
print(add.name, add.description, add.args, add.args_schema.schema())