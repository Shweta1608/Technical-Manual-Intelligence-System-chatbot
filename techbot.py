import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from groq import Client  


groq_api_key = os.getenv("groq_api_key")
GROQ_MODEL_NAME="llama-3.3-70b-versatile"

DB_FAISS_PATH="vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

from langchain_groq import ChatGroq

def load_llm(groq_model_name,groq_api_key):
    llm = ChatGroq(
        model=groq_model_name,
        temperature=0.5,
        max_tokens=512,  
        groq_api_key=groq_api_key  
    )
    return llm

def main():
    st.title("Ask Techbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                "You are Techbot to assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question "
                "If you don't know the answer, say that please contact MProfit support at support@mprofit.in"
                "Use three sentences maximum and keep the answer concise."
                
                Context: {context}
                Question: {question}
                
                """

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(groq_model_name=GROQ_MODEL_NAME, groq_api_key=groq_api_key),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            #source_documents=response["source_documents"]
            result_to_show=result#+"\nSource Docs:\n"#+str(source_documents)
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
            
    

