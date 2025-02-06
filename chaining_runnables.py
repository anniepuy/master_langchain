"""
Title: LangChain Chaining Runnables
Purpose: Learn how to take two chains in sequence but also run a third as parallel to the first two chains
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

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

#chains include a template, invoke, and LLM into one runnable script. The template and LLM get connected as one object for the invoke 
#Taking the output of one chain and adding it to another
system = SystemMessagePromptTemplate.from_template("You are a {school} teacher. You answer in short sentences.")
question = HumanMessagePromptTemplate.from_template("tell me about the {topics} in {points} points.")

messages = [system,question]

template = ChatPromptTemplate(messages)

#Create chain
chain = template|llm | StrOutputParser()
response = chain.invoke({'school': 'Middle school', 'topics': 'Mars', 'points': 4})
print("Response with StrOutputParser: ", response)

#we are going to pass this chain to another chain . the first LLM generates a response, and the second LLM evaluates the response
#anlaysis  - output from original chain is passed to this chain
analysis_prompt = ChatPromptTemplate.from_template(''' analyse the the following text: {response}
                                                   You need to tell me how difficult is it to understand.
                                                   Answer in one sentence only.
                                                   ''')

fact_check_chain = analysis_prompt | llm | StrOutputParser()
output = fact_check_chain.invoke({'response': response})
print(output)

#next mix the two chains together
composed_chain = {"response": chain} | analysis_prompt | llm | StrOutputParser()
final_output = composed_chain.invoke({'school': 'Middle school', 'topics': 'Mars', 'points': 4})
print(final_output)