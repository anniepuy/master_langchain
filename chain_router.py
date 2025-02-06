"""
Title: LangChain Chain Router
Purpose: If you have specific LLM's that you want to use for specific parts of the query, you can split the query.
Example: You have a general LLM, FinTECH LLM, and Medicine LLM. The chain router will split the query to which LLM best fits the use case.
The below uses a general LLM, a that will either decide if the query is Positive or Negative. The router will send positives to a positive chain and a negative to negative chain
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from dotenv import load_dotenv
import os
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

prompt = """Given the user review below, classifiy it as either being about `Positive` or `Negative`.
        Do not respond with more than one word.

        Review: {review}
        Classification:"""

template = ChatPromptTemplate.from_template(prompt)

chain = template | llm | StrOutputParser()

#review = "Thank you so much for providing such a great platform for leaning. I am really happy with the service."
review = "I am not happy with the service. I will not be using this service again."

result = chain.invoke({'review': review})
#test the output
#print(result)

#Next create the chain for positive reviews
positive_prompt = """
    You are expert in writing reply for positive reviews.
    You need to encourage the user to share their experiences on social media.
    Review: {review}
    Answer:"""

positive_template = ChatPromptTemplate.from_template(positive_prompt)
positive_chain = positive_template | llm | StrOutputParser()

##Next create the chain for negative reviews 
negative_prompt = """
    You are a customer service representative whose speciality is handling negative reviews.
    You are replying directly to the customer.
    You need first apologize to the user for the inconvenience.
    You need to encourage the user to share their concern through emailing: 'help@help.com'
    Review: {review}
    Answer:"""

negative_template = ChatPromptTemplate.from_template(positive_prompt)
negative_chain = negative_template | llm | StrOutputParser()

#Write a function to route the review to the correct chain
def review_route(info):
    if 'positive' in info['sentiment'].lower():
        return positive_chain
    else:
        return negative_chain
    
#final template
#print(review_route({'sentiment': 'positive'}))
#print(review_route({'sentiment': 'negative'}))

#need lamba function to route the review to the correct chain
from langchain_core.runnables import RunnableLambda
full_chain = {"sentiment": chain, 'review': lambda x:x['review']} | RunnableLambda(review_route) | llm | StrOutputParser()

output = full_chain.invoke({'review': review})
print(output)