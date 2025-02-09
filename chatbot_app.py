"""
Title: Creating own chatbot using StreamLit and Ollama
Purpose: Basic Chatbot with StreamLit with history enabled
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ( 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,   
    ChatMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()

#base set up
base_url = "http://localhost:11434"

model = 'llama3.2'
user_id = "user1"

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 0.8,
)

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_memory.db")

#Base LLM SetUp
system = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")

messages = [system, MessagesPlaceholder(variable_name='history'), human]
prompt = ChatMessagePromptTemplate(messages= messages)

chain = prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, input_message_key='input', history_messages_key='history')