"""
Title: YouTube Transcript SEO generator
Purpose: Learn how to load transcripts from YouTube and generate SEO-friendly content. Very dependent on the video selected - not all videos will work. Had to pic a smaller vidoe under 20 min to work on M4 Pro.
Author: Ann Hagan - via learning through Laxmi Kant on Udemy
"""

from langchain_community.document_loaders import YoutubeLoader
from scripts import llm

#Youtube Link
url = "https://www.youtube.com/watch?v=T_KZaJ744c4"

#loader = YoutubeLoader.from_youtube_url(url)

#docs = loader.load()
#print(docs)

#doc = docs[0]
#print(doc.metadata)

#doc.page_content
#length per tockens if we are doing 4 tokens per character - this would be 17000 tokens
#content is TOO large, teh LLM forgets content in between. 
#print(len(doc.page_content))

#to solve LLM forgetfulness, breakdown the data into smaller chunks.
#Read small chunks of the video
from langchain_community.document_loaders.youtube import TranscriptFormat

loader = YoutubeLoader.from_youtube_url(url, transcript_format=TranscriptFormat.CHUNKS, chunk_size_seconds=600)

docs = loader.load()
#print(len(docs2))
#print(docs2[1].metadata)
#print(docs2[1].page_content)
doc= docs[0]
doc.page_content

#multistate LLM. Broad keyword first, then narrow down to specifics
question = """You are an assitant for generating SEO keywords for YouTube.
            Please generate a list of keywords from the above content.
            You can use your creativity and correct spelling if needed."""
keywords = []
for doc in docs:
    response = llm.ask_llm(context=doc.page_content, question=question)
    keywords.append(response)

print(keywords)

