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

load_dotenv()  # for local dev; on Streamlit Cloud use st.secrets
mlflow.set_experiment("rag-deep-research")

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
                st.success("✅ Files uploaded and index rebuilt successfully!")
            except Exception as e:
                st.error(f"❌ Error during ingestion: {e}")

        st.info("You can now ask questions about the newly uploaded documents.")

st.markdown("---")


#  Chat interface




st.subheader("💬 Chat with your documents")

# New: difficulty level for this question
difficulty = st.selectbox(
    "Select difficulty of this question:",
    ["easy", "medium", "hard"],
    index=0,
)

if "history" not in st.session_state:
    st.session_state.history = []

def clear_input():
    st.session_state["user_input"] = ""

user_q = st.text_input("Your question:", key="user_input")

col1, col2 = st.columns([1, 5])
with col1:
    ask_btn = st.button("Ask", type="primary")
with col2:
    clear_btn = st.button("Clear chat", on_click=clear_input)

if clear_btn:
    st.session_state.history = []

#if ask_btn and user_q.strip():
   # st.session_state.history.append(("You", user_q))
    #try:
      #  rag_chain = build_chain()  # build using latest index
     #   answer = rag_chain.invoke(user_q)
    #except Exception as e:
    #    answer = f"⚠️ Error running RAG chain: {e}"

   # st.session_state.history.append(("Bot", answer))
#
if ask_btn and user_q.strip():
    st.session_state.history.append(("You", user_q))

    start_time = time.time()
    answer = ""

    with mlflow.start_run(run_name=f"rag-query-{difficulty}"):
        # ── Params: searchable info ──────────────────────────────
        mlflow.log_params({
            "difficulty": difficulty,
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-large",
            # short preview to keep the param table readable
            "question_preview": user_q[:200],
        })

        try:
            rag_chain = build_chain()  # uses latest index
            answer = rag_chain.invoke(user_q)
            latency = time.time() - start_time

            # ── Metrics: numbers to compare ──────────────────────
            mlflow.log_metric("latency_s", latency)
            mlflow.log_metric("answer_len", len(answer))

            # ── Artifacts: full text of interaction ─────────────
            mlflow.log_text(user_q, "question.txt")
            mlflow.log_text(answer, "answer.txt")

            # Optional: combined JSON artifact
            mlflow.log_dict(
                {
                    "question": user_q,
                    "difficulty": difficulty,
                    "answer": answer,
                    "latency_s": latency,
                },
                "interaction.json",
            )

        except Exception as e:
            answer = f"⚠️ Error running RAG chain: {e}"
            mlflow.log_param("error", str(e))

    st.session_state.history.append(("Bot", answer))

for who, msg in st.session_state.history:
    if who == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
        st.markdown("---")
