import os
import streamlit as st
import time

import ingest  # reuse existing ingest.main()
import mlflow

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
DB_DIR = "chroma_db"
# ─── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
DB_DIR = "chroma_db"

load_dotenv()  # harmless for local dev

# --- Databricks MLflow config via Streamlit secrets only ---

# Required secrets
DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

# Tracking & experiment
MLFLOW_TRACKING_URI = st.secrets.get("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_EXPERIMENT_NAME = st.secrets.get("MLFLOW_EXPERIMENT_NAME", "rag-deep-research")

# Make sure Databricks plugin sees these
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# IMPORTANT: set tracking URI BEFORE experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Optional: debug in sidebar (no secrets)
st.sidebar.markdown("### MLflow config (debug)")
st.sidebar.write(f"Tracking URI: {mlflow.get_tracking_uri()}")
st.sidebar.write(f"Experiment: {MLFLOW_EXPERIMENT_NAME}")

# UI #

st.set_page_config(page_title="RAG Deep Research", page_icon="💬")

# Custom styling: light blue background, white chat bar
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d6ecff 0%, #f0f9ff 100%);
}

/* White chat bar */
div[data-baseweb="input"] > div {
    background-color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #cccccc !important;
}
button[kind="primary"] {
    border-radius: 10px !important;
    padding: 0.25rem 1.25rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("RAG Deep Research 💬")
st.caption("Upload documents, build an index, and ask questions about them.")


#  RAG chain helper

SYSTEM = """You are a precise research assistant.
Use ONLY the provided context from the documents.
If the context is insufficient, say you don't know.
Cite chunks as [doc:#]."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations.")
])

def format_docs(docs):
    lines = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown").split("/")[-1]
        lines.append(f"[{src} | doc:{i}] {d.page_content}")
    return "\n\n".join(lines)

@st.cache_resource(show_spinner=False)
def get_llm():
    # On Streamlit Cloud, use st.secrets instead of os.getenv
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    # NOTE: keep model_name synced with what you log to MLflow below
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

def build_chain():
    """Build a fresh RAG chain using the current Chroma DB."""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    vs = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    llm = get_llm()

    rag_chain = (
        RunnableParallel({"docs": retriever, "question": RunnablePassthrough()})
        | {
            "context": lambda x: format_docs(x["docs"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

#  Document uploader + indexing

st.subheader("📄 Upload documents")

os.makedirs(DATA_DIR, exist_ok=True)

uploaded_files = st.file_uploader(
    "Upload PDF or text files to add to the knowledge base",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"{len(uploaded_files)} file(s) selected.")
    for f in uploaded_files:
        st.write("•", f.name)

index_btn = st.button("Upload & Re-index", type="primary")

if index_btn:
    if not uploaded_files:
        st.warning("Please select at least one file to upload before re-indexing.")
    else:
        with st.spinner("Uploading files and rebuilding index..."):
            # Save uploaded files into ./data
            for uploaded in uploaded_files:
                save_path = os.path.join(DATA_DIR, uploaded.name)
                with open(save_path, "wb") as out_f:
                    out_f.write(uploaded.getbuffer())

            # Run your ingestion pipeline to rebuild chroma_db
            try:
                ingest.main()
                st.cache_resource.clear()  # clear cached LLM if needed

                # Optional: log a re-index event to MLflow
                try:
                    with mlflow.start_run(run_name="reindex", nested=True):
                        mlflow.log_param("event_type", "reindex")
                        mlflow.log_param("n_uploaded_files", len(uploaded_files))
                        mlflow.log_param(
                            "uploaded_filenames",
                            ", ".join([f.name for f in uploaded_files])
                        )
                except Exception:
                    # Don't break the app if logging fails
                    pass

                st.success("✅ Files uploaded and index rebuilt successfully!")
            except Exception as e:
                st.error(f"❌ Error during ingestion: {e}")

        st.info("You can now ask questions about the newly uploaded documents.")

st.markdown("---")

#  Chat interface (modern chat UI)

st.subheader("💬 Chat with your documents")

# Move difficulty to sidebar so the main area feels more like a chat app
difficulty = st.sidebar.selectbox(
    "Question difficulty:",
    ["easy", "medium", "hard"],
    index=0,
)

# Button to clear the whole conversation
if st.sidebar.button("Clear chat"):
    st.session_state.pop("messages", None)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Sticky input bar at the bottom of the page
user_q = st.chat_input("Ask something about your documents...")

if user_q:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_q})

    with st.chat_message("user"):
        st.markdown(user_q)

    # --- MLflow logging around the RAG call ---
    start_time = time.time()
    try:
        rag_chain = build_chain()  # build using latest index

        with mlflow.start_run(run_name="chat-query"):
            # Params
            mlflow.log_param("user_query", user_q)
            mlflow.log_param("difficulty", difficulty)
            mlflow.log_param("model_name", "gpt-4o-mini")

            # Invoke RAG chain
            answer = rag_chain.invoke(user_q)

            # Metrics
            latency = time.time() - start_time
            mlflow.log_metric("latency_sec", latency)

            # Artifact: answer text
            mlflow.log_text(str(answer), "artifacts/answer.txt")

    except Exception as e:
        answer = f"⚠️ Error running RAG chain: {e}"

    # Add assistant message to history and display
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
