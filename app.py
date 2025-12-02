import os
import time
import streamlit as st

import ingest
import mlflow

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="RAG Deep Research", page_icon="💬")

DATA_DIR = "data"
DB_DIR = "chroma_db"

load_dotenv()  # used locally; Streamlit Cloud uses st.secrets


# -----------------------------------------------------------------------------
# Databricks + MLflow Config
# -----------------------------------------------------------------------------
DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

MLFLOW_TRACKING_URI = st.secrets.get("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_EXPERIMENT_NAME = st.secrets.get("MLFLOW_EXPERIMENT_NAME", "rag-deep-research")

os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# -----------------------------------------------------------------------------
# Sidebar: Title, Difficulty, Document Upload/Browse, Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("RAG Deep Research 💬")
    st.caption("Chat with your document-aware assistant. Test data can be downloaded from project repository, or upload your own.")

    st.markdown("### Question Settings")
    difficulty = st.selectbox(
        "Difficulty:",
        ["Easy", "Medium", "Hard"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Document Management")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Upload documents
    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="sidebar_upload",
    )

    if st.button("Re-index documents", type="primary", key="sidebar_reindex"):
        if not uploaded_files:
            st.warning("Please upload at least one file before re-indexing.")
        else:
            with st.spinner("Uploading & rebuilding index..."):
                for upl in uploaded_files:
                    save_path = os.path.join(DATA_DIR, upl.name)
                    with open(save_path, "wb") as f:
                        f.write(upl.getbuffer())

                try:
                    ingest.main()
                    st.cache_resource.clear()

                    with mlflow.start_run(run_name="reindex", nested=True):
                        mlflow.log_param("event_type", "reindex")
                        mlflow.log_param("n_uploaded_files", len(uploaded_files))
                        mlflow.log_param(
                            "files",
                            ", ".join([f.name for f in uploaded_files]),
                        )

                    st.success("Index rebuilt successfully!")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

    st.markdown("### Files in Index")
    existing_files = os.listdir(DATA_DIR)
    if existing_files:
        st.write("Indexed files:")
        for x in existing_files:
            st.write("•", x)
    else:
        st.write("No documents indexed yet.")

    st.markdown("---")
    st.markdown("### Session Controls")
    if st.button("Clear chat history"):
        st.session_state.pop("messages", None)

    st.markdown("---")
    st.markdown("### MLflow Debug")
    st.write("Tracking URI:", mlflow.get_tracking_uri())
    st.write("Experiment:", MLFLOW_EXPERIMENT_NAME)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d6ecff 0%, #f0f9ff 100%);
}
button[kind="primary"] {
    border-radius: 10px !important;
    padding: 0.25rem 1.25rem !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# RAG Chain Helper
# -----------------------------------------------------------------------------
SYSTEM = """You are a precise research assistant.
Use ONLY the provided context from the documents.
If the context is insufficient, say you don't know.
Cite chunks as [doc:#]."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations.")
])

def format_docs(docs):
    out = []
    for i, d in enumerate(docs):
        name = d.metadata.get("source", "unknown").split("/")[-1]
        out.append(f"[{name} | doc:{i}] {d.page_content}")
    return "\n\n".join(out)

@st.cache_resource(show_spinner=False)
def get_llm():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

def build_chain():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    vs = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    llm = get_llm()

    return (
        RunnableParallel({"docs": retriever, "question": RunnablePassthrough()})
        | {
            "context": lambda x: format_docs(x["docs"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# -----------------------------------------------------------------------------
# Main Chat UI
# -----------------------------------------------------------------------------
st.subheader("Chat with your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

st.markdown(
    "<div style='font-size: 0.85rem; color: #555;'>Ask a question below.</div>",
    unsafe_allow_html=True,
)

# Sticky input
user_q = st.chat_input("Type your question here...")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # RAG + MLflow
    start = time.time()
    try:
        rag_chain = build_chain()

        with mlflow.start_run(run_name="chat-query"):
            mlflow.log_param("user_query", user_q)
            mlflow.log_param("difficulty", difficulty)
            mlflow.log_param("model_name", "gpt-4o-mini")

            answer = rag_chain.invoke(user_q)

            latency = time.time() - start
            mlflow.log_metric("latency_sec", latency)

            mlflow.log_text(str(answer), "artifacts/answer.txt")

    except Exception as e:
        answer = f"⚠️ Error running RAG chain: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
