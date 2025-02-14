"""
Title: resume_parsing_streamlit.py
Purpose: Resume Parsing with Streamlit
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""
from dotenv import load_dotenv
load_dotenv(".env")

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


base_url = "http://localhost:11434"
model = 'llama3.1'

llm = ChatOllama(base_url=base_url, model=model, temperature=1)

#1.load resumes
filename = "Resume-1.pdf"
loader = PyMuPDFLoader('resume/{}'.format(filename))

docs = loader.load()
#print(docs[0].page_content)

##Pass this information to the LLM but first, format it
context = docs[0].page_content

question = """You are tasked with parsing a job resume. Your goal is to extract relevant information in a valid JSON structure format.
Do not write preambles or explanations."""

#Helper functions - Could put this in a different script for simplicity, leaving here
system = SystemMessagePromptTemplate.from_template("""You are a helpful assistant who answers questions based on resumes in valide JSON structure.""")

prompt = """
        **Task:** Extract key information from the followin resume text.

        ** Resume Text: **
        {context}

        **Instructions:**
        Please extract the following information and format it in a clear structure:

        1. **Contact Information:**
        - Name:
        - Email:
        - Phone Number:
        - Website/Portfolio:

        2. **Education:**
        - Institution Name:
        - Degree:
        - Field of Study:
        - Graduation Year:

        3. **Experience:**
        - Job Title:
        - Companhy Name:
        - Location:
        - Dates of Employment:
        - Responsibilities/Projects:

        4. **Projects:**
        - Project Title:
        - Description/Technologies Used:
        - Outcomes/Results:

        5. **Skills:**
        - Programming Languages:
        - Technologies/Tools:

        6. ** Additional Information:**
        - Certifications:
        - Awards or Honors:
        - Languages:

        **Question:**
        {question}

        **Extracted Information:**
        """

prompt = HumanMessagePromptTemplate.from_template(prompt)

def ask_llm(context,question):
    messages = [system, prompt]
    template = ChatPromptTemplate(messages)
    qna_chain = template | llm | StrOutputParser()
    return qna_chain.invoke({'context': context, 'question': question})

def validate_json(data):
    json_prompt = """
                Please validate and correct the following JSON data:

                **Extracted Information:**
                {data}

                Provide only the corrected JSON, with no preambles or explanations.
                ***Corrected JSON:***  """
    json_prompt = HumanMessagePromptTemplate.from_template(json_prompt)
    json_messages = [system, json_prompt]
    json_template = ChatPromptTemplate(json_messages)
    json_chain = json_template | llm | RunnablePassthrough()
    return json_chain.invoke({'data': data})

response = ask_llm(context=context, question=question)
#print(response)

##Validate JSON
validate_json_response = validate_json(response)
print(validate_json_response)
