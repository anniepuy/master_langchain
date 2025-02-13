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

db.run(response)

##Final Query Chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)
sql_query = create_sql_query_chain(llm, db)

final_chain = (
    {'context': sql_query, 'question': RunnablePassthrough()}
    | get_correct_sql_query
    | execute_query
)

question = "how many employees are there?"
response = final_chain.invoke(question)
print(response)

#3. Create Agent using LangGragh - tools prebuilt earlier are already in LangChain tooklits
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()
print(tools)

#create the prompt
from langchain_core.messages import SystemMessage

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
given an input question, create a systactically correct SQLite query to run, then 
look at the results of the query and return the answer.  Unless the user specifies a specific number
of examples they wish to obtain, always limit your query to at most 5 results.  You can order the results by a relevant column to return the 
most interesting examples in the database.  Never query for all the columns from a specific table, only ask for the relevent
columns givent the question.  You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it.  If you get an error while executing a query, rewrite the query and try agian.

DO NOT make any DML statement (INSERT, UPDATE, DELETE, DROP, etc.) to the database.
To start you shouold ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevent tables."""

system_message = SystemMessage(content=SQL_PREFIX)

##Now create the agent from langgraph
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools, state_modifier=system_message, debug=True)

question = "how many employees are there?"
response = agent_executor.invoke({'messages': [HumanMessage(content=question)]})
print(response)