"""
Title: Personal Health Agent
Purpose: Agent that serves as a personal health assistant
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()

llm = ChatOllama(model='llama3.2', base_url="http://localhost:11434")