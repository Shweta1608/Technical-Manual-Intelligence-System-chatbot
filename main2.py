from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
#from dotenv import load_dotenv
from groq import Client  # Correct import

# Load environment variables
#load_dotenv()

# Get Groq API key
groq_api_key = os.getenv("groq_api_key")
GROQ_MODEL_NAME="llama-3.3-70b-versatile"

# if not groq_api_key:
#     raise ValueError("Groq API key is not set in the .env file")

# print("Successfully loaded Groq API key.")

from langchain_groq import ChatGroq

def load_llm(groq_model_name):
    llm = ChatGroq(
        model=groq_model_name,
        temperature=0.5,
        max_tokens=512,  # Adjust as needed
        groq_api_key=groq_api_key  # Ensure you set this securely
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question "
    "If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "dont provide any source document information , make sure that you provide answer upto 3 lines only"
Context: {context}
Question: {question}

"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(GROQ_MODEL_NAME),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
#print("SOURCE DOCUMENTS: ", response["source_documents"])