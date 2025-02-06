from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

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

#chains include a template, invoke, and LLM into one runnable script. The template and LLM get connected as one object for the invoke 
#Squencial LCEL Chain
system = SystemMessagePromptTemplate.from_template("You are a {school} teacher. You answer in short sentences.")
question = HumanMessagePromptTemplate.from_template("tell me about the {topics} in {points} points.")

messages = [system,question]

#old method to add variables.
# next grab template
template = ChatPromptTemplate(messages)

print(template)

#next make the question
question = template.invoke({'school': 'elementary', 'topics': 'sun', 'points': 4})

#pass the question to llm
response = llm.invoke(question) 
print(response.content)

#Create chain
chain = template|llm
print(f"Chain: {chain}")

#now you can pass all of the variables in the chain
response2 = chain.invoke({'school': 'elementary', 'topics': 'sun', 'points': 4})
print(response2.content)
#print metadata with the model and response - input and output tokens
print(response2.usage_metadata)

#To just view the content only as a string not anything else. use output parsing to reduce what is outputted
#stroutputparser() - only returns the content as a string with no metadata
from langchain_core.output_parsers import StrOutputParser
#Create chain
chain2 = template|llm | StrOutputParser()
response3 = chain2.invoke({'school': 'Middle school', 'topics': 'Mars', 'points': 4})
print("Response with StrOutputParser: ", response3)