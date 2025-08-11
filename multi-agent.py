import os
import getpass
import faiss
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode, tools_condition


class MultiState(TypedDict):
  messages: Annotated[list, add_messages]

vectorstore = None

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = init_chat_model("google_genai:gemini-2.0-flash")


def setup_pdfs( pdf_folder: str = "pdfs" ):
  global vectorstore
  loader = DirectoryLoader(pdf_folder, glob="**/*.pdfs", loader_cls=PyPDFLoader)
  docs = loader.load()
  chunks = splitter.split_documents(docs)
  vectorstore = FAISS.from_documents(chunks, embedding)

  return vectorstore

setup_pdfs()