import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


OPENAI_API_KEY = "your key here"


# Step_1: Read .pdf files
# import streamlit as st

# Creating web interface to upload PDF files:
st.header("My first chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type='pdf')

# Step_2: Extract uploaded text and break in into chunks
# from PyPDF2 import PdfReader

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Breaking loaded text into smaller chunks so GPT can read it
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # generating embeddings
    # from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Creating vector store - using FAISS (Facebook AI Semantic Search)
    # from langchain.vectorstores import FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user question
    user_question = st.text_input("Type your question here")

    # Similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)

        # Defining our LLM <<- this is fine-tuning part!!!
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        # Input results - we need to create a chain of sequences
        # from langchain.chains.question_answering import load_qa_chain

        # Take the question -> get relevant document -> pass it to LlM -> generate output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)
