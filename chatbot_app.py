"""
Title: Creating own chatbot using StreamLit and Ollama
Purpose: Basic Chatbot with StreamLit with history enabled through session state only. Not persistent.
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
user_id = st.text_input("Enter your user ID", "ann")

llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 0.8,
)

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_memory.db")

#Create chat history in streamlist session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#Button for user history
if st.button("Start new conversation"):
    #Clear the chat history of streamlist
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    #clear history of mysql db
    history.clear()


#display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


#Base LLM SetUp
system = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")

messages = [system, MessagesPlaceholder(variable_name='history'), human]
prompt = ChatPromptTemplate.from_messages(messages)

chain = prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, input_message_key='input', history_messages_key='history')

#Method for chatting with LLM
def chat_with_llm(session_id, input):
    #non stream method
    #output = runnable_with_history.invoke({'input': input}, config={'configurable': {'session_id': session_id}})
    #return output

    #stream method
    for output in runnable_with_history.stream({'input': input}, config={'configurable': {'session_id': session_id}}):
        yield output

#input chat message box
prompt = st.chat_input("Ask your question here")

#if a user has entered a prompt, update the chat history
if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ##non stream response
    #response = chat_with_llm(user_id, prompt)
    
    #with st.chat_message("assistant"):
        #st.markdown(response)

    #streamling response
   
    #writes the responseon the screen
    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm(user_id, prompt))
    ## add the response to the chat history
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
    