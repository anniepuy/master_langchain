"""
Title: Creating own chatbot using StreamLit and Ollama
Purpose: Basic Chatbot with StreamLit with history enabled
Note: Replaced ChatMessagePromptTemplate with ChatPromptTemplate
	•	ChatMessagePromptTemplate is meant for individual messages.
	•	ChatPromptTemplate.from_messages(messages) is needed when handling multiple messages.
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
To run: streamlit run chatbot_app.py
"""
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ( 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,   
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()

##Streamlit HTML Element
st.title("Chatbot with History")
st.write("This is a simple chatbot with history enabled. It uses the Ollama model to generate responses.")

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
prompt = ChatPromptTemplate.from_messages(messages)

chain = prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, input_message_key='input', history_messages_key='history')