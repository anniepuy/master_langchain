from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama

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

#make the sysstem message once, and pass the variables through the system message
#you need Langchain Prompt Templates to do this. 
from langchain_core.prompts import ( 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate
)

#Add variables inside the question and system prompt
system = SystemMessagePromptTemplate.from_template("You are a {school} teacher. You answer using anaologies.")
question = HumanMessagePromptTemplate.from_template("tell me about the {topics} in {points} points.")

print(system)
print(question)

#pass the variables inside the prompt
#print(question.format( topics= 'DoD Acquisition Process', points = 4))
#print(system.format( school= 'high school'))

#put it all together 
messages = [system, question]
template = ChatPromptTemplate(messages)

final_question = template.invoke({'school': 'ph.d', 'topics': 'sun', 'points': 4})
response = llm.invoke(final_question)
print(response.content)