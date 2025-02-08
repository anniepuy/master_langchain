"""
Title: LangChain JSON Output Parser
Purpose: Learn how to structure the LLM output through using the output parsers.
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    PromptTemplate
)
from langchain_core.output_parsers import JsonOutputParser

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

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

#create pydantic class - the LLM sends AI messages and then to Pydantic. Extend the BaseModel with custom method.
#in this example, the pydantic class will be a method that validates a Joke.
class Joke(BaseModel):
        """
        Joke to tell user
        """
        #Description must be self explanatory for LLM
        #must pass each three of these b/c they become parameters
        setup: str = Field(description="The set up of the joke.")
        punchline: str = Field(description="The punchline of the joke.")
        rating: Optional[int] = Field(description="The rating of the joke is from 1 to 10.", default=None)

parser = JsonOutputParser(pydantic_object=Joke)
print(parser.get_format_instructions())

prompt = PromptTemplate(
        template = '''
        Answer the user query with a joke. Here is your formatting instruction.
        {format_instruction}
        Query: {query}
        Answer: ''',
        input_variables=['query'],
        partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = prompt | llm | parser

output = chain.invoke({'query': 'Tell me a joke'})
print("\n\n")
print(output)