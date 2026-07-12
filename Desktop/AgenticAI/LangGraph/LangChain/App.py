import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = Ollama(model="llama3.1")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question very carefully ensure that it feels like written by human"),
    ("user", "the question is {question}")
])

st.title("Basic Chat Application")

input_txt=st.text_input("Ask anything...")

outputParser=StrOutputParser()

final_answer=prompt|llm|outputParser

if input_txt:
    st.write(final_answer.invoke(input_txt))





