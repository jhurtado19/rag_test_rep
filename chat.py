import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_DIR = "chroma_db"

# Swap these three lines if using Azure OpenAI:
# from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
# llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini", temperature=0)
# embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vs = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vs.as_retriever(search_kwargs={"k": 4})

SYSTEM = """You are a precise, citation-minded assistant.
Use ONLY the provided context to answer.
If unsure or context is missing, say you don't know.
Cite chunks as [doc:#] where # is the chunk index.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "User question: {question}\n\nContext:\n{context}\n\nAnswer with citations.")
])

def format_docs(docs):
    # include a simple index to cite: [doc:i]
    return "\n\n".join([f"[doc:{i}] {d.page_content}" for i, d in enumerate(docs)])

# RAG chain: retrieve -> prompt -> LLM -> text
rag_chain = (
    RunnableParallel({"docs": retriever, "question": RunnablePassthrough()})
    | {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- API ----
app = FastAPI()

class Ask(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Ask):
    answer = rag_chain.invoke(q.question)
    return {"answer": answer}

# Run: uvicorn chat:app --reload