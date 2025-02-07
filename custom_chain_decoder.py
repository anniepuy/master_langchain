"""
Title: LangChain Custom Chain Decoder
Purpose: When you make a custom chain with chain decoder, do not use the 'chain' decorator.
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""
from langchain_core.runnables import chain
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

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
#Creating two separate chains
#always need system and question message - this section here is used all the time.
system = SystemMessagePromptTemplate.from_template("You are a {school} teacher. You answer in short sentences.")
question = HumanMessagePromptTemplate.from_template("tell me about the {topics} in {points} points.")

messages = [system,question]

template = ChatPromptTemplate(messages)

#Create chain
fact_chain = template|llm | StrOutputParser()

#create the next chain - that creates a poem. Add a new variable  
question = HumanMessagePromptTemplate.from_template("write a poem on {topics} in {sentences} lines.")

messages = [system,question]

template2 = ChatPromptTemplate(messages)

#Create chain
poem_chain = template2|llm | StrOutputParser()

#Create a method
def custom_chain(params):
    return{
        'fact': fact_chain.invoke(params),
        'poem': poem_chain.invoke(params),
    }

#Convert the method to a RunnableLambda custom chain
params = {'school': 'Middle school', 'topics': 'Mars', 'points': 4, 'sentences': 4}
custom_output = custom_chain(params)
print('n\n')
print(custom_output)