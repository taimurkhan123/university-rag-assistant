import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from ingest import download_drive_folder, build_faiss_index, GDRIVE_FOLDER_ID, LOCAL_DATA_DIR, FAISS_DB_PATH

# 1. Page Configuration
st.set_page_config(
    page_title="UniGuide AI | University Assistant",
    page_icon="🎓",
    layout="centered"
)

# 2. Modern UI CSS Styling
custom_css = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        text-align: center;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    [data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Header Setup
st.markdown('<div class="main-title">🎓 UniGuide AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Instant, accurate details on fees, admissions, courses & campus life</div>', unsafe_allow_html=True)

# 4. API Key Verification (Groq key starts with "gsk_")
grok_api_key = st.secrets.get("GROK_API_KEY") or os.getenv("GROK_API_KEY")

if not grok_api_key:
    st.error("🔑 Groq API key is missing! Please configure GROK_API_KEY in Streamlit Secrets.")
    st.stop()

# 5. Document Ingestion & Index Sync
if not os.path.exists(FAISS_DB_PATH):
    with st.status("🚀 Syncing Google Drive documents & building vector index...", expanded=True) as status:
        download_drive_folder(GDRIVE_FOLDER_ID, LOCAL_DATA_DIR)
        build_faiss_index()
        status.update(label="✅ Index built successfully!", state="complete", expanded=False)

# 6. Load FAISS Vector Store (Cached)
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)

vector_db = load_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 7. Initialize Groq LLM (OpenAI-compatible endpoint)
llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",          # Groq-supported model
    openai_api_key=grok_api_key,              # Your "gsk_..." key
    openai_api_base="https://api.groq.com/openai/v1",  # Groq endpoint
    temperature=0.2
)

system_prompt = (
    "You are an expert AI assistant for university affairs.\n"
    "Answer the student's question clearly and concisely using ONLY the retrieved context below.\n"
    "If the answer is not present in the context, clearly state that you don't know.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 8. Chat History Render
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 9. User Input Processing
if user_query := st.chat_input("Ask about tuition, admissions, hostels..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing university records..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)
            
            with st.expander("🔍 View Retrieved Sources"):
                for doc in response["context"]:
                    st.markdown(f"**Document Source:** `{doc.metadata.get('source', 'Drive File')}`")
                    st.caption(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})
