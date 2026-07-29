import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
import tempfile

load_dotenv()

st.set_page_config(page_title="RAG Research Assistant", layout="wide")
st.title("Multi-Source RAG Research Assistant")
st.markdown("Upload PDFs or enter URLs — ask questions with source-backed answers")

tab1, tab2 = st.tabs(["PDF Upload", "Web URL"])

docs = []

with tab1:
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs.extend(loader.load())
            os.unlink(tmp.name)
        st.success(f"Loaded {len(uploaded_files)} PDF(s)")

with tab2:
    urls = st.text_area("Enter URLs (one per line)")
    if urls:
        url_list = [u.strip() for u in urls.strip().split("\n") if u.strip()]
        loaded_count = 0
        for url in url_list:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
                loaded_count += 1
            except Exception:
                st.error(f"Failed to load: {url}")
        st.success(f"Loaded {loaded_count} URL(s)")

if docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # --- Rate-limit-safe batched embedding ---
    # Free tier for gemini-embedding-001 allows ~100 embedding requests/minute.
    # Sending all chunks in one call (Chroma.from_documents) blows past that
    # instantly on any non-trivial document set, causing a 429 RESOURCE_EXHAUSTED
    # error. We batch the chunks and pace requests, with retry/backoff as a
    # safety net for any batch that still gets rate-limited.
    #
    # This whole build is also cached (@st.cache_resource) so it runs ONCE per
    # uploaded file set, not on every Streamlit rerun (e.g. every time you type
    # a question). Without caching, asking N questions would re-embed the same
    # chunks N times and burn through the quota for no reason.

    BATCH_SIZE = 20  # stay well under the 100/min free-tier cap
    BATCH_DELAY_SECONDS = 20  # pacing between batches

    @retry(
        retry=retry_if_exception_type(ResourceExhausted),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        stop=stop_after_attempt(10),
    )
    def add_batch_with_retry(vs, batch):
        vs.add_documents(batch)

    @st.cache_resource(show_spinner=False)
    def build_vectorstore(_chunks, _file_signature):
        # _file_signature forces the cache to invalidate when a different
        # file/URL set is uploaded, even though _chunks itself isn't hashable
        first_batch = _chunks[:BATCH_SIZE]
        vs = Chroma.from_documents(first_batch, embeddings)

        for i in range(BATCH_SIZE, len(_chunks), BATCH_SIZE):
            batch = _chunks[i:i + BATCH_SIZE]
            time.sleep(BATCH_DELAY_SECONDS)
            add_batch_with_retry(vs, batch)

        return vs

    # Lightweight signature so the cache knows when the underlying files changed
    file_signature = tuple(
        (d.metadata.get("source", ""), len(d.page_content)) for d in chunks
    )

    est_seconds = (len(chunks) // BATCH_SIZE) * BATCH_DELAY_SECONDS
    with st.spinner(f"Embedding {len(chunks)} chunks (first run only, ~{est_seconds}s)..."):
        vectorstore = build_vectorstore(chunks, file_signature)

    st.success(f"Vector store ready ({len(chunks)} chunks)")

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.3, 0.7]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=hybrid_retriever,
        return_source_documents=True
    )

    st.divider()
    query = st.text_input("Ask a question about your documents:")

    if query:
        with st.spinner("Searching..."):
            result = qa.invoke(query)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Answer")
            st.write(result["result"])

        with col2:
            st.subheader("Sources Used")
            for i, doc in enumerate(result["source_documents"][:3]):
                with st.expander(f"Source {i+1}"):
                    st.caption(f"From: {doc.metadata.get('source', 'Unknown')}")
                    st.write(doc.page_content[:300] + "...")

        if st.button("Export Q&A as Report"):
            report = f"Q: {query}\n\nA: {result['result']}\n\n---\nSources:\n"
            for doc in result["source_documents"][:3]:
                report += f"- {doc.metadata.get('source', 'Unknown')}\n  {doc.page_content[:200]}...\n\n"
            st.download_button("Download Report", report, file_name="rag_report.txt")
else:
    st.info("Upload PDFs or add URLs to begin")