"""
Title: LangChain Custom Chain Router
Purpose: If you have specific LLM's that you want to use for specific parts of the query, you can split the query.
Example: you want to add a custom chain function. These are called RunnableLambda. This allows the RunnablePassThrough to go to a different or specific LLM
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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

#Create a custom method
def char_counts(text):
    return len(text)

def word_counts(text):
    return len(text.split())

#Prepare chatprompt template- this will be passed to the first LLM which generates a suboutput as the input to custom function
prompt = ChatPromptTemplate.from_template("Explain these inputs in 3 points: {input1} and {input2}")

#Convert the method to a RunnableLambda chain
#this will only return the char counts and word counts
chain = prompt | llm | StrOutputParser() | {'char_counts': RunnableLambda(char_counts), 'word_counts': RunnableLambda(word_counts)}
output = chain.invoke({'input1': 'Earth is planet', 'input2': 'Sun is star'})
print(output)

#THis will return all three outputs
chain2 = prompt | llm | StrOutputParser() | {'char_counts': RunnableLambda(char_counts), 'word_counts': RunnableLambda(word_counts), 'output': RunnablePassthrough()}

output2 = chain2.invoke({'input1': 'Earth is planet', 'input2': 'Sun is star'})
print(output2)
