"""
Title: LangChain Chat Memory
Purpose: Learn how to add memory to your LLM chatbot. After you build the chain, the save response is triggered based on session ID & user ID.
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
Note: does not work - need to return to research and solve later
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ( 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,   
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

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

'''
It wraps a Runnable and manages the chat history before the Runnable is invoked.
clas enables multiple conversations by saving each conversation with a session_id
Not using system prompt templates because it is using history. Unless you want the LLM to remember the system role.
'''

def get_sesssion_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_memory.db")

system = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant.")

human = HumanMessagePromptTemplate.from_template("{input}")

messages = [system, MessagesPlaceholder(variable_name='history'), human]

prompt = ChatPromptTemplate(messages= messages)

chain = prompt | llm | StrOutputParser()

##Prepare Runnable with message hist
runnable_with_history = RunnableWithMessageHistory(chain, get_sesssion_history, input_message_key='input',
                                              history_message_key='history')


def chat_with_llm(session_id, input):
    output = runnable_with_history.invoke({'input': input}, 
                                          config={'configurable': {'session_id': session_id}})
    return output


user_id = 'user1'
print(chat_with_llm(user_id, 'What is the capital of France?'))
