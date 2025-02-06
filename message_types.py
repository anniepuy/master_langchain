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

question = "tell me about the earth in three points."

# using system message - the role of the LLM response can be dynamically changed instead of hardcoding just one type of model (example, shirlock holmes vs normal)
from langchain_core.messages import SystemMessage, HumanMessage

#pass the question inside the HumanMessage
human_question = HumanMessage(question)
#system = SystemMessage('You are an elementary teacher. You answer in short sentences.')
#system = SystemMessage('You are an high school teacher. You answer using real world examples.')
#system = SystemMessage('You are a yoga teacher. You answer is based on the Yamas and Niyamas.')
#system = SystemMessage('You are a Catholic priest. You answer is based on knowledge of the bible.')
#system = SystemMessage('You are a Navy pilot. You answer is based on knowledge planetary sciences.')
system = SystemMessage('You are a Navy pilot. You answer is based on military knowledge and MGRS.')


messages = [system, human_question]
response = llm.invoke(messages)
print(response.content)