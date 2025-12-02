import os
import time
import streamlit as st

import ingest  # reuse existing ingest.main()
import mlflow

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── Basic Config ────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Deep Research", page_icon="💬")

DATA_DIR = "data"
DB_DIR = "chroma_db"

load_dotenv()  # mainly for local dev, Streamlit Cloud uses st.secrets

# ─── Databricks + MLflow Config ──────────────────────────────────────────

# Required secrets for Databricks-backed MLflow
DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

# Tracking & experiment
MLFLOW_TRACKING_URI = st.secrets.get("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_EXPERIMENT_NAME = st.secrets.get("MLFLOW_EXPERIMENT_NAME", "rag-deep-research")

# Ensure MLflow / Databricks SDK see these in the environment
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# IMPORTANT: set tracking URI BEFORE experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# ─── Sidebar UI (static title, settings, debug) ──────────────────────────

with st.sidebar:
    st.title("RAG Deep Research 💬")
    st.caption("Chat with your document-aware assistant.")

    st.markdown("---")
    st.subheader("Question Settings")

    difficulty = st.selectbox(
        "Difficulty of this question:",
        ["easy", "medium", "hard"],
        index=0,
    )

    if st.button("Clear chat"):
        st.session_state.pop("messages", None)

    st.markdown("---")
    st.subheader("MLflow (debug)")
    st.write(f"Tracking URI: {mlflow.get_tracking_uri()}")
    st.write(f"Experiment: {MLFLOW_EXPERIMENT_NAME}")

# ─── Custom Styling (background, minor tweaks) ───────────────────────────

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

# ─── RAG Chain Helper ────────────────────────────────────────────────────

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
    # On Streamlit Cloud, use st.secrets; locally, env vars are fine
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
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

# ─── Main: Chat + Upload / Re-index ──────────────────────────────────────

st.subheader("Chat with your documents")

# Initialize chat history for the modern chat UI
if "messages" not in st.session_state:
    # messages: list of {"role": "user" | "assistant", "content": str}
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Upload & Re-index SECTION (below messages, above chat input)
st.markdown("---")
st.subheader("Upload & Re-index")

os.makedirs(DATA_DIR, exist_ok=True)

uploaded_files = st.file_uploader(
    "Upload PDF or text files to add to the knowledge base",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

reindex_btn = st.button("Upload & Re-index", type="primary", key="reindex_main")

if reindex_btn:
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

                # Log re-index event to MLflow (optional)
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

st.markdown(
    "<div style='font-size: 0.85rem; color: #555; margin-top: 0.5rem;'>"
    "Ask a question about your indexed documents below."
    "</div>",
    unsafe_allow_html=True,
)

# Sticky chat input at the bottom of the page
user_q = st.chat_input("Type your question here...")

if user_q:
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # 2. Run RAG chain + log to MLflow
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

    # 3. Add and display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
