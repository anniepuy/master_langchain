## Replacement for LangSmith

Opik
Langfuse

## Organization of the Scripts

### Basic Ollama Set up

script.py

### Message Types

message_types.py

### Prompt Templates

prompt_templates.py

### Basic LangChain Chain

sequential_chain.py
custom_chain.py
custom_chain_decoder.py
chain_router.py
chaining_runnables.py
creating_parallel_chains_1.py

### Output parsing

csv_output_parsing.py
date_time_parsing.py
json_output_parser.py
output_parsing.py

### Chat Message History

chat_memory.py

## Full LLM with Streamlit

chatbot_app.py

## PDF Parsing

pdf_parsing.py

## Using PDF Parsing in Chains with LLM

pdf_summary.py
rag_chat_qa.py

## Creating structured Markdown Report

generate_structured_report.py

## Webscraping & Chunking

webpage_loaders.py
chunking_with_webscraping.py

## Single LLM script - standalone

llm_qa_script.py

## Structured Outputs

PPTX_data_loader.py - includes script output for speech
XML_structured_output.py - Includes output with Markdown
personalized_email.phy - Doc uploader with personalized email output

## Load YouTube Video Transcript and Create SEO

youtube_seo.py

## Full RAG Scripts

Filed under rag_with_vector folder

# embedding set up with FAIS and nomi-embed-text:

embedding_faiss_setup_standalone.py

# Different searches using LangChain : simularity, MMR, simularity with threshold

simularity_searches_with_faiss.py

# final RAG pipeline with Runnables

rag_with_ollama.py

## Tool calling for AI Agents

Phi3 models do not support tool calling - Llama does
