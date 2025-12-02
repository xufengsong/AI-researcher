import streamlit as st
import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv(".env")
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 만약 .env에 OPENAI_API_BASE가 있다면 주석 처리하거나 제거해야 공식 API가 호출됩니다.
# os.environ.pop("OPENAI_API_BASE", None) 

# ---- DOC LOAD ----
def load_and_split_docs(uploaded_file):
    # 임시 파일로 저장하여 로더가 읽을 수 있게 함
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(uploaded_file.name)
    else:
        loader = TextLoader(uploaded_file.name, encoding="utf-8")

    documents = loader.load()
    # 청크 사이즈를 조금 더 키우고 오버랩을 넉넉히 주는 것이 문맥 유지에 유리할 수 있음
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    
    # 처리 후 임시 파일 삭제 (선택 사항)
    # os.remove(uploaded_file.name)
    
    return docs

# ---- VECTOR STORE ----
def get_vectorstore(docs):
    # text-embedding-3-small 모델 사용
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# ---- RAG CHAIN ----
def build_rag_chain(vectordb):
    retriever = vectordb.as_retriever()
    
    prompt = ChatPromptTemplate.from_template("""
    너는 반도체 기술 문서를 기반으로 답변하는 AI야.
    아래 질문에 주어진 [참고 문서]의 내용만을 바탕으로 답해.
    문서에 없는 내용은 "문서에서 찾을 수 없습니다"라고 말해.

    질문:
    {question}

    [참고 문서]:
    {context}
    """)

    # [수정] 모델명을 gpt-4.1-mini -> gpt-4o-mini 로 변경
    llm = ChatOpenAI(
        model=" gpt-4.1-mini", 
        temperature=0
    )

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever,
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt
        | llm
    )
    return rag_chain

# ---- UI ----
st.set_page_config(page_title="반도체 문서 RAG 챗봇")
st.title("📘 반도체 기술문서 RAG 챗봇")

if "uploaded_file_hash" not in st.session_state:
    st.session_state.uploaded_file_hash = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

uploaded_file = st.file_uploader("문서를 업로드하세요 (PDF, TXT)", type=["pdf", "txt"])

def get_file_hash(bytes_data):
    return hashlib.md5(bytes_data).hexdigest()

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)

    if st.session_state.uploaded_file_hash != file_hash:
        with st.spinner("문서 분석 및 임베딩 생성 중..."):
            try:
                split_docs = load_and_split_docs(uploaded_file)
                st.session_state.vectordb = get_vectorstore(split_docs)
                st.session_state.rag_chain = build_rag_chain(st.session_state.vectordb)
                st.session_state.uploaded_file_hash = file_hash
                st.success("문서 처리가 완료되었습니다!")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

    # 파일이 업로드되고 처리가 완료된 상태에서만 질문 입력창 표시
    if st.session_state.rag_chain:
        question = st.text_input("질문을 입력하세요:")

        if question:
            with st.spinner("답변 생성 중..."):
                response = st.session_state.rag_chain.invoke({"question": question})
                st.write(response.content)