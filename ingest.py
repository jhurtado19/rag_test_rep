import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"


def load_docs():
    loaders = [
        DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
    ]
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading docs with {loader}: {e}")
    return docs


def main():
    docs = load_docs()
    if not docs:
        print("No documents found in 'data' — nothing to ingest.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key,
    )

    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print(f"Ingested {len(chunks)} chunks into {DB_DIR}")


if __name__ == "__main__":
    main()
