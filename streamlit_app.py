import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv("env.txt")  # 환경 변수 로드 (API Key 등)

# 문서 처리 관련 라이브러리
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 모델 및 임베딩
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 체인 구성 요소
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

langfuse = get_client()

# Replacement for old CallbackHandler()
langfuse_handler = CallbackHandler()



# -------------------------------
# 문서 처리 및 체인 구성 (캐시로 한 번만 실행)
@st.cache_resource(show_spinner="체인을 초기화하는 중입니다...")
def create_chain():
    # 1. PDF 로딩
    loader = PyMuPDFLoader("data/아이디어 보호를 위한 가이드라인 개정 해설서.pdf")
    docs = loader.load()

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    split_docs = text_splitter.split_documents(docs)

    # 3. 벡터 저장소 생성
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_base=os.environ["EMBED_BASE_URL"]
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. 프롬프트 및 모델
    prompt = ChatPromptTemplate.from_template(
        """
        다음은 문서에서 검색된 정보입니다:
        {context}

        이 정보를 바탕으로 사용자의 질문에 대해 답하되, 반드시 다음 원칙을 지키세요.
        1. 위 문서의 내용만 근거로 답변하세요.
        2. 답변에는 반드시 해당 내용의 조항, 항목과 페이지 정보를 함께 명시하세요. (예: "3.2항, p.8")
        3. 문서에 관련 정보가 없으면 "정보 없음"이라고만 답하세요.
        4. 허구의 규정, 금액, 조항번호 등을 절대 만들어내지 마세요.
        5. 답변은 간결하고, 불필요한 해설 없이 한두 문장으로만 작성하세요.

        예시 질문 : 외부인에게 아이디어를 공유해도 되나요?
        예시 답변 : 가이드라인 3.4항에 따라, 비밀유지계약(NDA) 체결 후 공유 가능합니다. (출처: 3.4항, p.9)

        질문: {question}
        """
    )

    llm = ChatOpenAI(model="openai/gpt-4.1-mini", temperature=0)
    parser = StrOutputParser()

    # LCEL 기반 체인 구성
    chain = (
        RunnableLambda(
            lambda x: {
                "context": retriever.invoke(x["question"]),
                "question": x["question"],
            }
        )
        | prompt
        | llm
        | parser
    )

    return chain


# -------------------------------
# Streamlit UI 구성
st.set_page_config(
    page_title="📄 아이디어 보호를 위한 가이드라인 챗봇"
)
st.title("📄 아이디어 보호 가이드라인 챗봇")

# 체인 초기화
chain = create_chain()

# 입력창
user_input = st.text_input(
    "질문을 입력하세요:",
    placeholder="예: 제안서에 기밀 표시를 안 하면 보호받을 수 있나요?",
)

# 응답 출력
if user_input:
    with st.spinner("답변 생성 중입니다..."):
        try:
            response = chain.invoke(
                {"question": user_input},
                config={"callbacks": [langfuse_handler]}
            )
            st.success(response)
        except Exception as e:
            st.error(f"에러 발생: {str(e)}")
