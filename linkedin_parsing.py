"""
Title: linkedin_parsing.py
Purpose: LinkedIN Profile Parsing using Selenium
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
#requires pip install selenium & beautifulsoup4
"""

from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import ChatOllama
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

load_dotenv()

#this opens a Chrome webbrowser
driver = webdriver.Chrome()

driver.get("https://www.linkedin.com/login")

driver.title
#print(driver.title)

email = driver.find_element(By.ID, "username")
email.send_keys('LINKEDIN_EMAIL')
password = driver.find_element(By.ID, "password")
password.send_keys('LINKEDIN_PASSWORD')
password.submit()

url="https://www.linkedin.com/in/YOUR LINKEDIN URL/"
driver.get(url)

#load the source - all the sections. This page section is TOO large for ChatGPT - so we need to strip out uncessary sections
page_source = driver.page_source

#preprocessing to support LLM limitations of large data
#converts it to HTML object
soup = BeautifulSoup(page_source, 'html.parser')

#limits the result to just the text - but this is STILL too large for ChatGPT
soup.get_text()

#to reduce the data, we look at it per sections
profile = soup.find('main', {'class':"JISPKWUUlCCBuwzAaWeDmonfWqcesafwG"})

#print(profile.get_text())
sections = profile.find_all('section', {'class':"artdeco-card"})

#print(len(sections))
#get each text from the 14 sections
sections_text = [sections.get_text() for sections in sections]

#clean up the sections text using regex
import re

#remove mulitple new lines and tabs
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\t+', '\t', text)
    text = re.sub(r'\t\s+', ' ', text)
    text = re.sub(r'\n\s+', '\n', text)
    return text

#apply clean text to section text
sections_text_clean = [clean_text(section) for section in sections_text]

#now remove duplicates
def remove_duplicates(text):
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if line[:len(line)//2] == line[len(line)//2:]:
            new_lines.append(line[:len(line)//2])
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

sections_text_clean_no_duplicates = [remove_duplicates(section) for section in sections_text_clean]

#final clean data

#apply clean _data to LLM
from langchain_core.prompts import(SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser

base_url = "http://localhost:11434"
model = 'llama3.2'
llm =ChatOllama(base_url=base_url, model=model)

system = SystemMessagePromptTemplate.from_template(
    """You are a helpful assistant who answer LinkedIN Profile parsing related to user question based on the provided profile text data.""")

def ask_llm(prompt):
    prompt = HumanMessagePromptTemplate.from_template(prompt)

    messages = [system, prompt]
    template = ChatPromptTemplate(messages)

    qna_chain = template | llm | StrOutputParser()

    return qna_chain.invoke({})

template = """
Extract and return the requested information from the LinkedIn profile data in a concise, point-by-point format(up to 5 points). Avoid any preambles or advertisements.

### LinkedIn Profile Data:
{}

### Information to Extract:
Extract '{}' in bullet points, limited the output to 5 points. Provide only the necessary details.
Remember it is a LinkedIn profile data.

### Extracted Information:"""

context = sections_text[0]
k = "Name and Headline"

#string formatting template
prompt = template.format(context, k)
response = ask_llm(prompt)


#Parse Data section wise
section_keys = ['Name and Headline']
for section in sections_text[1:]:
    section_keys.append(section.strip().split('\n')[0])
section_keys

#sections text
responses = {}

for k, context in zip(section_keys, sections_text):
    prompt = template.format(context, k)
    response = ask_llm(prompt)
    responses[k] = response

#save as JSON
import json
with open('linkedin_data.json', 'w') as f:
    json.dump(responses, f, indent=4)