'''
Purpose: Learn how to take two chains in sequence but also run a third as parallel to the first two chains'''

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
output = fact_chain.invoke({'school': 'Middle school', 'topics': 'solar system', 'points': 2})

#create the next chain - that creates a poem. Add a new variable  
question = HumanMessagePromptTemplate.from_template("write a poem on {topics} in {sentences} lines.")

messages = [system,question]

template2 = ChatPromptTemplate(messages)

#Create chain
poem_chain = template2|llm | StrOutputParser()
output_poem = poem_chain.invoke({'school': 'middle school', 'topics': 'solar system', 'sentences': 4})
print(output_poem)

#Creating a third chain that runs the two chains in parallel. The output is two keys - fact and poem
from langchain_core.runnables import RunnableParallel

#combine fact and poem chain together with keys for the next chain
runnable_chain = RunnableParallel(fact=fact_chain, poem=poem_chain)

output_runnable = runnable_chain.invoke({'school': 'Middle school', 'topics': 'Mars', 'points': 4, 'sentences': 4})
print(output_runnable)
print(output_runnable['fact'])
print('\n\n')
print(output_runnable['poem'])