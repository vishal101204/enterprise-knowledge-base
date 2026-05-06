import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import (
    RetrievalQA, ConversationalRetrievalChain, LLMChain, MapReduceDocumentsChain,
    StuffDocumentsChain, ReduceDocumentsChain
)
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_classic.agents import AgentType, initialize_agent
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
import tempfile, os, json
from datetime import datetime

st.set_page_config(page_title="Enterprise Knowledge Hub", page_icon="🏢", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;500&display=swap');
* { font-family: 'Outfit', sans-serif; }
.stApp { background: #060b14; color: #e2e8f0; }
h1 { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.8rem; font-weight: 900; letter-spacing: -2px;
}
h2, h3 { color: #94a3b8; font-weight: 600; }
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 1.5rem; font-weight: 600;
    transition: all 0.3s ease; font-size: 0.95rem;
}
.stButton > button:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 10px 30px rgba(102,126,234,0.4);
}
.module-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1));
    border: 1px solid rgba(102,126,234,0.3); border-radius: 12px;
    padding: 1.2rem; margin: 0.5rem 0; transition: all 0.3s ease;
}
.module-card:hover { border-color: rgba(102,126,234,0.6); transform: translateX(4px); }
.answer-box {
    background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.3);
    border-radius: 12px; padding: 1.5rem; color: #e2e8f0;
    line-height: 1.7;
}
.summary-box {
    background: rgba(240,147,251,0.08); border: 1px solid rgba(240,147,251,0.3);
    border-radius: 12px; padding: 1.5rem; color: #e2e8f0; margin: 1rem 0;
}
.chat-user {
    background: rgba(102,126,234,0.15); border-radius: 12px 12px 4px 12px;
    padding: 1rem; margin: 0.5rem 0; color: #e2e8f0;
}
.chat-ai {
    background: rgba(240,147,251,0.08); border-radius: 12px 12px 12px 4px;
    padding: 1rem; margin: 0.5rem 0; color: #e2e8f0;
    border-left: 3px solid #764ba2;
}
.stat-chip {
    background: rgba(102,126,234,0.2); border: 1px solid rgba(102,126,234,0.4);
    border-radius: 20px; padding: 0.3rem 0.8rem; font-size: 0.8rem;
    color: #a5b4fc; display: inline-block; margin: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}
.nav-tab { cursor: pointer; transition: all 0.2s; }
label { color: #94a3b8 !important; }
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(102,126,234,0.3) !important;
    border-radius: 8px !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(102,126,234,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🏢 Enterprise Knowledge Hub")
st.markdown("#### *Full-Stack LangChain + RAG Platform with AI Agents, Summarization & Semantic Search*")

# Sidebar
with st.sidebar:
    st.markdown("## 🔑 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model = st.selectbox("LLM Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.markdown("---")
    st.markdown("### 📁 Knowledge Base")
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    web_url = st.text_input("Or load from URL:", placeholder="https://example.com/article")
    chunk_size = st.slider("Chunk Size", 300, 2000, 800)
    chunk_overlap = st.slider("Chunk Overlap", 0, 300, 100)
    st.markdown("---")
    st.markdown("### 🧠 Memory")
    memory_k = st.slider("Conversation Window (k)", 2, 20, 5)
    st.markdown("---")
    if st.button("🗑️ Reset Knowledge Base"):
        for key in ["vectorstore", "chat_history", "memory", "doc_stats", "summaries"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Reset!")

# State init
defaults = {
    "vectorstore": None,
    "chat_history": [],
    "doc_stats": {},
    "summaries": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=memory_k, memory_key="chat_history", return_messages=True, output_key="answer"
    )

# Navigation tabs
tab1, tab2, tab3, tab4 = st.tabs(["📥 Knowledge Base", "💬 Chat & Q&A", "📊 Document Summary", "🤖 AI Agent"])

# TAB 1: Knowledge Base
with tab1:
    st.markdown("## 📥 Build Your Knowledge Base")
    st.markdown("Upload documents from multiple sources to create your enterprise knowledge graph.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class='module-card'>
        <h4 style='color:#a5b4fc'>📄 File Ingestion</h4>
        <p style='color:#94a3b8;font-size:0.9rem'>Upload PDFs, Word documents, and text files. All content is chunked, embedded, and stored in ChromaDB for semantic retrieval.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='module-card'>
        <h4 style='color:#c4b5fd'>🌐 Web Ingestion</h4>
        <p style='color:#94a3b8;font-size:0.9rem'>Load and index content directly from web URLs. The system scrapes, cleans, and vectorizes web content automatically.</p>
        </div>""", unsafe_allow_html=True)

    sources = []
    if uploaded_files:
        sources.extend([f.name for f in uploaded_files])
    if web_url:
        sources.append(web_url)

    if sources:
        st.markdown(f"**Sources to index:** " + " ".join([f"<span class='stat-chip'>{s[:30]}</span>" for s in sources]), unsafe_allow_html=True)

    if (uploaded_files or web_url) and api_key:
        if st.button("⚡ Build Enterprise Knowledge Base", use_container_width=True):
            with st.spinner("Ingesting and indexing all sources..."):
                try:
                    all_chunks = []
                    doc_stats = {}

                    if uploaded_files:
                        for file in uploaded_files:
                            suffix = "." + file.name.split(".")[-1]
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(file.read())
                                tmp_path = tmp.name
                            if suffix == ".pdf": loader = PyPDFLoader(tmp_path)
                            elif suffix == ".txt": loader = TextLoader(tmp_path)
                            elif suffix == ".docx": loader = Docx2txtLoader(tmp_path)
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["source_name"] = file.name
                                doc.metadata["source_type"] = "file"
                                doc.metadata["ingested_at"] = datetime.now().isoformat()
                            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            chunks = splitter.split_documents(docs)
                            all_chunks.extend(chunks)
                            doc_stats[file.name] = {"chunks": len(chunks), "type": suffix[1:].upper(), "pages": len(docs)}
                            os.unlink(tmp_path)

                    if web_url:
                        loader = WebBaseLoader(web_url)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source_name"] = web_url
                            doc.metadata["source_type"] = "web"
                        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        chunks = splitter.split_documents(docs)
                        all_chunks.extend(chunks)
                        doc_stats[web_url] = {"chunks": len(chunks), "type": "WEB", "pages": len(docs)}

                    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    vectorstore = Chroma.from_documents(all_chunks, embeddings, persist_directory="/tmp/enterprise_kb")
                    st.session_state.vectorstore = vectorstore
                    st.session_state.doc_stats = doc_stats

                    st.success(f"✅ Knowledge base built! {len(all_chunks)} chunks across {len(doc_stats)} sources.")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.doc_stats:
        st.markdown("### 📊 Knowledge Base Statistics")
        cols = st.columns(len(st.session_state.doc_stats))
        for i, (name, stats) in enumerate(st.session_state.doc_stats.items()):
            with cols[i]:
                st.markdown(f"""<div class='module-card' style='text-align:center'>
                <div class='stat-chip'>{stats['type']}</div>
                <p style='color:#e2e8f0;margin:0.5rem 0;font-size:0.85rem'>{name[:25]}...</p>
                <h3 style='color:#a5b4fc;margin:0'>{stats['chunks']}</h3>
                <small style='color:#64748b'>chunks</small>
                </div>""", unsafe_allow_html=True)

# TAB 2: Chat Q&A
with tab2:
    st.markdown("## 💬 Conversational Q&A")
    st.markdown("Chat with your knowledge base using contextual memory.")

    if st.session_state.chat_history:
        for msg in st.session_state.chat_history[-10:]:
            css_class = "chat-user" if msg["role"] == "user" else "chat-ai"
            icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"<div class='{css_class}'>{icon} <strong>{msg['role'].title()}:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("Ask anything:", placeholder="What are the key findings across all documents?", label_visibility="collapsed")
    with col2:
        top_k = st.selectbox("K", [3, 5, 7, 10], index=1, label_visibility="collapsed")

    if query and st.session_state.vectorstore and api_key:
        with st.spinner("Retrieving from knowledge base..."):
            try:
                llm = ChatOpenAI(model=model, openai_api_key=api_key, temperature=temperature)
                system_template = """You are an expert enterprise knowledge assistant with access to a comprehensive document database.
Provide precise, well-structured answers citing specific documents when possible.
If information spans multiple documents, synthesize and mention all relevant sources.
Always be professional, accurate, and thorough.

Context from knowledge base:
{context}"""
                human_template = "Question: {question}\n\nPlease provide a comprehensive answer:"
                messages = [
                    SystemMessagePromptTemplate.from_template(system_template),
                    HumanMessagePromptTemplate.from_template(human_template)
                ]
                qa_prompt = ChatPromptTemplate.from_messages(messages)
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm, retriever=retriever, memory=st.session_state.memory,
                    return_source_documents=True, combine_docs_chain_kwargs={"prompt": qa_prompt}
                )
                result = qa_chain({"question": query})
                answer = result["answer"]
                sources = list(set([d.metadata.get("source_name", "Unknown") for d in result.get("source_documents", [])]))

                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                st.markdown(f"<div class='answer-box'><strong style='color:#a5b4fc'>🤖 Answer:</strong><br><br>{answer}</div>", unsafe_allow_html=True)
                if sources:
                    st.markdown("**Sources:** " + " ".join([f"<span class='stat-chip'>{s[:30]}</span>" for s in sources]), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    elif query and not st.session_state.vectorstore:
        st.warning("⚠️ Build the knowledge base first in the Knowledge Base tab!")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferWindowMemory(
            k=memory_k, memory_key="chat_history", return_messages=True, output_key="answer"
        )

# TAB 3: Document Summarization
with tab3:
    st.markdown("## 📊 AI Document Summarization")
    st.markdown("Generate executive summaries using LangChain's MapReduce chain.")

    summary_file = st.file_uploader("Upload a document to summarize", type=["pdf", "txt", "docx"], key="summarizer")
    summary_type = st.radio("Summary Type", ["Executive Summary", "Detailed Analysis", "Key Points & Insights", "SWOT Analysis"])

    if summary_file and api_key:
        if st.button("📝 Generate Summary", use_container_width=True):
            with st.spinner("Generating AI summary with MapReduce chain..."):
                try:
                    suffix = "." + summary_file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(summary_file.read())
                        tmp_path = tmp.name

                    if suffix == ".pdf": loader = PyPDFLoader(tmp_path)
                    elif suffix == ".txt": loader = TextLoader(tmp_path)
                    elif suffix == ".docx": loader = Docx2txtLoader(tmp_path)
                    docs = loader.load()

                    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    chunks = splitter.split_documents(docs)

                    llm = ChatOpenAI(model=model, openai_api_key=api_key, temperature=0.3)

                    type_instructions = {
                        "Executive Summary": "Write a concise executive summary highlighting the main purpose, key findings, and recommendations.",
                        "Detailed Analysis": "Provide a detailed analysis covering all major topics, data points, and conclusions.",
                        "Key Points & Insights": "Extract and list the most important points, insights, and actionable takeaways.",
                        "SWOT Analysis": "Analyze the content and present findings as a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)."
                    }

                    map_template = f"""Analyze this section of the document:
{{docs}}
{type_instructions[summary_type]}
Summary:"""
                    map_prompt = PromptTemplate.from_template(map_template)
                    map_chain = LLMChain(llm=llm, prompt=map_prompt)

                    reduce_template = f"""Combine these summaries into a final comprehensive {summary_type}:
{{docs}}
Final {summary_type}:"""
                    reduce_prompt = PromptTemplate.from_template(reduce_template)
                    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
                    combine_docs_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")
                    reduce_documents_chain = ReduceDocumentsChain(
                        combine_documents_chain=combine_docs_chain,
                        collapse_documents_chain=combine_docs_chain,
                        token_max=4000
                    )
                    map_reduce_chain = MapReduceDocumentsChain(
                        llm_chain=map_chain, reduce_documents_chain=reduce_documents_chain,
                        document_variable_name="docs", return_intermediate_steps=False
                    )
                    summary = map_reduce_chain.run(chunks[:10])  # limit for demo
                    st.session_state.summaries[summary_file.name] = summary
                    os.unlink(tmp_path)

                    st.markdown(f"<div class='summary-box'><strong style='color:#c4b5fd'>📊 {summary_type}:</strong><br><br>{summary}</div>", unsafe_allow_html=True)
                    st.download_button("⬇️ Download Summary", summary, file_name=f"summary_{summary_file.name}.txt")
                except Exception as e:
                    st.error(f"Error: {e}")

# TAB 4: AI Agent
with tab4:
    st.markdown("## 🤖 Enterprise AI Agent")
    st.markdown("An intelligent agent that combines your knowledge base with web search and reasoning.")

    agent_query = st.text_area("Agent Task:", placeholder="Analyze the key trends in my documents and compare with current industry standards from the web...", height=100)

    col1, col2 = st.columns(2)
    with col1:
        use_kb = st.checkbox("Use Knowledge Base", value=True)
    with col2:
        use_search = st.checkbox("Enable Web Search", value=False)

    if agent_query and api_key:
        if st.button("🚀 Run AI Agent", use_container_width=True):
            with st.spinner("Agent thinking and executing..."):
                try:
                    llm = ChatOpenAI(model=model, openai_api_key=api_key, temperature=0.3)
                    tools = []

                    if use_kb and st.session_state.vectorstore:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                        def search_kb(query):
                            docs = retriever.get_relevant_documents(query)
                            return "\n\n".join([d.page_content for d in docs])
                        tools.append(Tool(name="KnowledgeBase", func=search_kb, description="Search the enterprise knowledge base for relevant documents and information."))

                    if use_search:
                        search_tool = DuckDuckGoSearchRun()
                        tools.append(Tool(name="WebSearch", func=search_tool.run, description="Search the web for current information and industry trends."))

                    # If no tools available, use direct LLM
                    if not tools:
                        result_text = llm.predict(agent_query)
                        st.markdown(f"<div class='answer-box'><strong style='color:#a5b4fc'>🤖 Agent Response:</strong><br><br>{result_text}</div>", unsafe_allow_html=True)
                    else:
                        agent = initialize_agent(
                            tools=tools, llm=llm,
                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            verbose=False, max_iterations=5, handle_parsing_errors=True
                        )
                        result = agent.run(agent_query)
                        st.markdown(f"<div class='answer-box'><strong style='color:#a5b4fc'>🤖 Agent Response:</strong><br><br>{result}</div>", unsafe_allow_html=True)

                    st.markdown("**Tools used:** " + " ".join([f"<span class='stat-chip'>{t.name}</span>" for t in tools]) if tools else "", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
    elif agent_query and not api_key:
        st.warning("⚠️ Please enter your OpenAI API Key in the sidebar!")
