# --------------------------------------------------------------------------------------------------------------------------------
# Using Flask

# from flask import Flask, render_template, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFacePipeline
# from dotenv import load_dotenv
# import os
# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     pipeline,
#     BitsAndBytesConfig,
# )
# from src.prompt import *

# app = Flask(__name__)
# load_dotenv()

# # Set API keys
# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

# # Load embeddings
# embeddings = download_hugging_face_embeddings()

# # Setup vector store (Pinecone)
# index_name = "medicalbot"
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name, embedding=embeddings
# )
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # Load Mistral model
# model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# # Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# # Configure for 8-bit quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # Enable 8-bit loading
#     llm_int8_threshold=6.0,  # Default threshold
# )

# # Load model in 8-bit
# model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.1",
#     device_map=device,
#     quantization_config=bnb_config,
#     torch_dtype=torch.float16,
# )

# # Create pipeline
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=1024,
#     temperature=0.7,
#     do_sample=True,
#     return_full_text=False,
# )

# llm = HuggingFacePipeline(pipeline=pipe)

# # Define prompt (chat style)
# prompt = PromptTemplate.from_template(
#     "<s>[INST] " + system_prompt + "\n\nQuestion:\n{input} [/INST]"
# )


# # Create RAG chain
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# # Routes
# @app.route("/")
# def index():
#     return render_template("chat.html")


# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     response = rag_chain.invoke({"input": msg})
#     return str(response["answer"])


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080, debug=True)

# --------------------------------------------------------------------------------------------------------------------------------
# Using Chainlit

import chainlit as cl
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from dotenv import load_dotenv
import torch
import os
from src.prompt import system_prompt

# Load environment variables
load_dotenv()

# Set API keys
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Setup Pinecone retriever
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load mistral model with 8-bit quantization
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
prompt = PromptTemplate.from_template(
    "<s>[INST] " + system_prompt + "\n\nQuestion:\n{input} [/INST]"
)

# RAG chain setup
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Track last 4 conversation turns per session
conversation_history = []


@cl.on_message
async def main(message: cl.Message):
    user_input = message.content

    # Append user message
    conversation_history.append({"role": "user", "content": user_input})

    # Trim to last 4 messages
    if len(conversation_history) > 4:
        conversation_history.pop(0)

    # Prepare context string
    conversation_context = ""
    for msg in conversation_history[:-1]:  # exclude the current user message
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_context += f"{role}: {msg['content']}\n"

    # Inject conversation history into the prompt
    full_input = f"{conversation_context}User: {user_input}"

    response = rag_chain.invoke({"input": full_input})

    # Append assistant's reply
    conversation_history.append({"role": "assistant", "content": response["answer"]})

    # Send response
    await cl.Message(content=response["answer"]).send()
