"""
Title: MySQL Queries
Purpose: Use LLM to retrieve data from MySQL
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
#requires pip install pymysql
"""

from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

 #base set up
base_url = "http://localhost:11434"

model = 'llama3.2'

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 0.8,
    num_predict = 256,
)
#Db connection
db = SQLDatabase.from_uri(your passsword & MySQL URI)

db.dialect
db.get_usable_table_names()

db.run("SELECT * FROM employees LIMIT 5")
#1. SQL Chains 
sql_chain = create_sql_query_chain(db, llm)

#prints ready to go prompts
sql_chain.get_prompts()[0].pretty_print()

print(sql_chain)

#this cannot correctly call tools or prompt
question = "how many employees are there?"
response = sql_chain.invoke(question)
print(response)

##Corrected Query from LLM
#import our own script we created earlier
from scripts.llm import ask_llm
from langchain_core.runnables import chain

@chain
def get_correct_sql_query(input):
    context = input['context']
    question = input['question']
    
    instruction = """
                Use above context to fetch the correct SQL query for following question
                {}
                
                Do not enclose query in ```sql and do not write preamble and explanation.
                You MUST return only single SQL query.""".format(question)
    response = ask_llm(context=context, question = question)

    return response

get_correct_sql_query.invoke({'context': response, 'question': question})
