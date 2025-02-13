"""
Title: LangChain CSV Output Parser
Purpose: Learn how to structure the LLM output through using the output parsers.
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    PromptTemplate
)
from langchain_core.output_parsers import CommaSeparatedListOutputParser

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

parser = CommaSeparatedListOutputParser()
print(parser.get_format_instructions())

format_instruction = parser.get_format_instructions()

prompt = PromptTemplate(
    template='''
    Answer the user query with a list of values. Here is your formatting instruction.
    {format_instruction}
    
    Query: {query}
    Answer: ''',
    input_variables=['query'],
    partial_variables={'format_instruction': format_instruction}
)

chain  = prompt | llm | parser

output = chain.invoke({'query': 'List the names of the planets in the solar system'})

print(output)

