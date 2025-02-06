from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama

load_dotenv()

#test loading the environment variables
print(os.environ["LANGCHAIN_ENDPOINT"])


base_url = "http://localhost:11434"

model = 'llama3.2'


llm = ChatOllama(
    base_url=base_url,
    model = model,
    temperature = 0.8,
    num_predict = 256,
)

response = llm.invoke('How do I make an AI agent?')
print(response)

#print only the LLM response
print(response.content)

#print metadata with the model and response - input and output tokens
print(response.usage_metadata)

#to view all of the returns - vector matches from the LLM
#for chunk in llm.stream('how do I make an AI agent?'):
#   print(chunk)

#to view all of the returns - vector matches from the LLM
#response2= ""
#for chunk in llm.stream('how do I make an AI agent?'):
   # response2 = response2 + " " + chunk.content
   # print(response)