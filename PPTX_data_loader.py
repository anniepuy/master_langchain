"""
Title: LangChain PPTX Data Loader
Purpose: Learn how to load Microsoft PowerPoint data into the LLM
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from langchain_community.document_loaders import UnstructuredPowerPointLoader

#Element loading does one page at a time
loader = UnstructuredPowerPointLoader("ms_data/ml_course.pptx", mode='elements')

docs = loader.load()
#print(len(docs))
#print(docs[0].metadata)
#read the documents page-wise through making a dictionary - combines all element
ppt_data = {}
for doc in docs:
    page = doc.metadata['page_number']
    ##if a page is empty, it will return an empty string 
    ppt_data[page] = ppt_data.get(page, "") + "\n\n" + doc.page_content

#print(ppt_data)

#prepare content slide -wise so it is easier to read - final output. Nice!!
context = ""
for page, content in ppt_data.items():
    context += f"### {page}:\n\n{content.strip()}\n\n"

#print(context)
#
##Write the LLM code
from scripts import llm_qa_script

##Prepare a question
question = """
For each PowerPoint slide provided above, write a 2-minute script from a that effectively conveys the key points. Ensure a smooth flow between slides, maintaining a clear and engaging narrative. """ 

response = llm_qa_script.ask_llm(context, question)
print(response)