import os
import getpass
import uuid
import sqlite3
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph
from typing import Annotated
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
load_dotenv()

file_path="./data"

loader = DirectoryLoader(file_path, glob="**/*.pdf", use_multithreading=True, loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
  chunk_size= 1000,
  chunk_overlap= 200,
  add_start_index= True
)

all_splite = splitter.split_documents(docs)

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_1 = embeddings.embed_query(all_splite[0].page_content)
vector_2 = embeddings.embed_query(all_splite[1].page_content)

assert len(vector_1) == len(vector_2)

# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

vector_store = Chroma(
  collection_name="collection_name",
  embedding_function=embeddings,
  persist_directory="./chroma_langgraph"
)

_ = vector_store.add_documents(documents=all_splite)

class State(TypedDict):
  question: str
  context: List[Document]
  answer: str
  messages: Annotated[List[BaseMessage], "conversation messages"]

def retrieve(state: State):
  retrieve_docs = vector_store.similarity_search(state["question"])
  return {"context": retrieve_docs}

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Question: {question}
Context: {context}
Answer:"""
prompt = PromptTemplate.from_template(template)

def generate(state: State):
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = prompt.invoke({"question": state["question"], "context": docs_content})
  response = llm.invoke(messages)
  return {"answer": response.content}

graph_builder = StateGraph(State)
graph_builder.add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")

os.makedirs("./memory", exist_ok=True)
conn = sqlite3.connect("./memory/memory.db", check_same_thread=False)
memory = SqliteSaver(conn)
store = InMemoryStore()
graph = graph_builder.compile(checkpointer=memory, store=store)

thread_id = str(uuid.uuid4())

def stream_graph_update(user_input: str):
   config = {"configurable": {"thread_id": thread_id}}
   for event in graph.stream({"question": user_input, "messages": [HumanMessage(content=user_input)]}, config=config):
      for value in event.values():
         if 'answer' in value:
            print("Assistant: ", value['answer'])


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "by", "bye"]:
        print("Good Bye")
        break
    
    # Check if the input is not empty to avoid errors
    if user_input.strip():
        stream_graph_update(user_input)