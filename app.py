import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import warnings
import shutil

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["USER_AGENT"] = "rag-system/1.0"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

st.set_page_config(
    page_title="RAG System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .answer-box {
        background-color: black;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: black;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

PERSIST_DIRECTORY = "./chroma_db"
if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'source_name' not in st.session_state:
    st.session_state.source_name = None

@st.cache_resource
def initialize_models():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
        return embeddings, llm
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None

embeddings, llm = initialize_models()

def load_document(source: str, uploaded_file=None):
    try:
        if uploaded_file is not None:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            os.remove(temp_path)
            return documents
            
        elif source.endswith(".pdf"):
            if not os.path.exists(source):
                raise ValueError(f"PDF file not found: {source}")
            loader = PyPDFLoader(source)
            return loader.load()
            
    except Exception as e:
        raise Exception(f"Error loading document: {str(e)}")

def process_and_index(documents):
    if not documents:
        raise ValueError("No documents to process")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore, len(chunks)

def get_retriever():
    if not os.path.exists(PERSIST_DIRECTORY):
        return None
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    ).as_retriever(search_kwargs={"k": 5})

def generate_response(query: str):
    retriever = get_retriever()
    if retriever is None:
        return "No document loaded. Please load a document first.", []
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template="""Use the following context to answer the question.
Context: {context}
Question: {question}
Answer based strictly on the context provided above. If the answer cannot be found in the context, say "I cannot find this information in the provided context.":""",
                input_variables=["context", "question"],
            )
        }
    )
    
    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]

def summarize_document():
    retriever = get_retriever()
    if retriever is None:
        return "No document loaded."
    
    all_docs = retriever.get_relevant_documents("Provide a comprehensive summary of all content")
    
    if not all_docs:
        return "No content available to summarize."
    
    context = "\n\n".join([doc.page_content for doc in all_docs[:10]])  
    
    prompt = f"""Summarize the following content concisely, highlighting the main points and key information:

{context}

Summary:"""
    
    summary = llm.invoke(prompt).content
    return summary

with st.sidebar:
    st.title("Document Management")
    if st.session_state.vectorstore_loaded and st.session_state.source_name:
        st.success(f"Loaded: {st.session_state.source_name}")
    else:
        st.info("ℹNo document loaded")
    st.divider()
    st.subheader("Load New Document")
    source_type = st.radio(
        "Select source type:",
        ["Upload PDF"],
        label_visibility="collapsed"
    )
    source = None
    uploaded_file = None
    if source_type == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        if uploaded_file:
            source = uploaded_file.name
    if st.button("Load & Process Document", type="primary"):
        if source or uploaded_file:
            with st.spinner("Loading and processing document..."):
                try:
                    documents = load_document(source, uploaded_file)
                    vectorstore, num_chunks = process_and_index(documents)
                    st.session_state.vectorstore_loaded = True
                    st.session_state.source_name = source if source else uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"Successfully indexed {num_chunks} chunks!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please provide a source")
    st.divider()
    if st.button("Clear Database"):
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            st.session_state.vectorstore_loaded = False
            st.session_state.source_name = None
            st.session_state.chat_history = []
            st.success("Database cleared")
            st.rerun()
st.title("RAG Question Answering System")
st.markdown("Ask questions about your documents and get AI-powered answers!")
if not st.session_state.vectorstore_loaded:
    st.info("Please load a document from the sidebar to get started")
else:
    tab1, tab2, tab3 = st.tabs(["Ask Questions", "Document Summary", "Chat History"])
    with tab1:
        st.subheader("Ask a Question")
        query = st.text_input(
            "Your question:",
            placeholder="What is this document about?",
            key="question_input"
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("Ask", use_container_width=True)
        if ask_button and query:
            with st.spinner("Searching for answer..."):
                try:
                    answer, sources = generate_response(query)
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer,
                        "sources": sources
                    })
                    st.markdown("### Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    if sources:
                        st.markdown("### Sources")
                        for i, src in enumerate(sources, 1):
                            with st.expander(f"Source {i}"):
                                st.text(src.page_content[:500] + "..." if len(src.page_content) > 500 else src.page_content)
                                st.caption(f"Metadata: {src.metadata}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    with tab2:
        st.subheader("Document Summary")
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                try:
                    summary = summarize_document()
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    with tab3:
        st.subheader("Chat History")
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {chat['question'][:60]}..."):
                    st.markdown("**Question:**")
                    st.write(chat['question'])
                    st.markdown("**Answer:**")
                    st.write(chat['answer'])
                    if chat['sources']:
                        st.markdown("**Sources:**")
                        for j, src in enumerate(chat['sources'], 1):
                            st.caption(f"Source {j}: {src.page_content[:200]}...")
        else:
            st.info("No chat history yet. Start asking questions!")
st.divider()
st.caption("Built with Streamlit and LangChain | Powered by Google Gemini")