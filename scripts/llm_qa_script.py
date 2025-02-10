"""
Title: LLM QA Script
Purpose: LLM script so that you can read it from any notebook or file.
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

#create th eprompts and chain
#Prepare system prompt and prompt template for QA
system = SystemMessagePromptTemplate.from_template("""
                            You are a helpful AI assistant who answers user question based on provided content.
                            Do not answer in more than {words} words.
                            """)    

prompt = """ Answer user question based on the provided context only.  If you do not know the answer, just say "I don't know".
        ### Context: 
        {context}

        ### Question: 
        {question}

        ### Answer:"""

prompt = HumanMessagePromptTemplate.from_template(prompt)

messages = [system, prompt]
template = ChatPromptTemplate(messages)


qna_chain = template | llm | StrOutputParser()

#create a method
def ask_llm(context,question):
    return qna_chain.invoke({'context': context, 'question': question})