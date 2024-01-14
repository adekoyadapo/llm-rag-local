from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import torch 
import streamlit as st
import os
import time
import random

if "history" not in st.session_state:
    st.session_state.history = []

load_dotenv()

def load_model(
    model_path="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    max_new_tokens=512,
    content_length=400,
    temperature=0.9,
):
    """
    Load a locally downloaded model.

    Parameters:
        model_path (str): The path to the model to be loaded.
        model_type (str): The type of the model.
        max_new_tokens (int): The maximum number of new tokens for the model.
        temperature (float): The temperature parameter for the model.

    Returns:
        CTransformers: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        SomeOtherException: If the model file is corrupt.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    # Additional error handling could be added here for corrupt files, etc.

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        max_new_tokens=max_new_tokens,
        comtent_length=content_length,  # type: ignore
        temperature=temperature,  # type: ignore
    )

    return llm


prompt_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
The example of your response should be:

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Vector Database
persist_directory = "./db/rag" # Persist directory path

embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

if not os.path.exists(persist_directory):
    with st.spinner('🚀 Starting your bot.  This might take a while'):
        # Data Pre-processing
        pdf_loader = DirectoryLoader("./docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
        text_loader = DirectoryLoader("./docs/", glob="./*.txt", loader_cls=TextLoader)
        
        pdf_documents = pdf_loader.load()
        text_documents = text_loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=5)
        
        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        text_context = "\n\n".join(str(p.page_content) for p in text_documents)

        pdfs = splitter.split_text(pdf_context)
        texts = splitter.split_text(text_context)

        data = pdfs + texts

        print("Data Processing Complete")

        vectordb = Chroma.from_texts(data, embeddings, persist_directory=persist_directory)
        vectordb.persist()

        print("Vector DB Creating Complete\n")

elif os.path.exists(persist_directory):
    vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embeddings)
    
    print("Vector DB Loaded\n")

prompt = st.chat_input("Say something")

def create_retrieval_qa_chain(llm, prompt, vectordb):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.

    This function initializes a RetrievalQA object with a specific chain type and configurations,
    and returns this QA chain. The retriever is set up to return the top 3 results (k=3).

    Args:
        llm (any): The language model to be used in the RetrievalQA.
        prompt (str): The prompt to be used in the chain type.
        db (any): The database to be used as the retriever.

    Returns:
        RetrievalQA: The initialized QA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt

qa_prompt = (
        set_custom_prompt()
    )
# Quering Model
query_chain = create_retrieval_qa_chain(load_model(), qa_prompt, vectordb)

st.title("RAG Chat Assistant")

for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])


# prompt = st.chat_input("Say something")

if prompt:
    st.session_state.history.append({
        'role':'user',
        'content':prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('💡Thinking'):
        
        response = query_chain({"query": prompt})

        st.session_state.history.append({
            'role' : 'Assistant',
            'content' : response['result']
        })

        with st.chat_message("Assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = response['result']
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
