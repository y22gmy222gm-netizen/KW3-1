import streamlit as st
import psycopg2
from google import genai
import time

# --- 1. 앱 설정 및 스타일 ---
st.set_page_config(
    page_title="강원연구원 AI 테스트 [2026-03-15]",
    page_icon="🤖",
    layout="wide"
)

# 답변 글자 파란색 및 모바일 최적화 스타일 적용
st.markdown("""
    <style>
    .ai-answer { color: blue; white-space: pre-wrap; font-size: 11pt; }
    /* 모바일에서 버튼 등이 더 잘 보이도록 조정 */
    .stChatInput { bottom: 20px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 초기 설정 및 세션 상태 관리 ---
DB_CONFIG = {
    "host": st.secrets["db_host"],
    "database": st.secrets["db_name"],
    "user": st.secrets["db_user"],
    "password": st.secrets["db_password"],
    "port": st.secrets["db_port"]
}
API_KEY = st.secrets["gemini_api_key"]

if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=API_KEY)
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. 핵심 로직 함수 ---
def process_ai_query(user_query):
    if user_query in st.session_state.response_cache:
        return st.session_state.response_cache[user_query], 0.0, True

    try:
        start_time = time.time()
        client = st.session_state.client
        
        # 1. 임베딩 (768-Slicing)
        embedding_response = client.models.embed_content(
            model="models/gemini-embedding-001", 
            contents=user_query,
            config={'task_type': 'retrieval_query'}
        )
        query_embedding = embedding_response.embeddings[0].values[:768]

        # 2. DB 검색
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            search_query = """
                SELECT content, file_name, page_num 
                FROM KW3_paper_sections 
                ORDER BY embedding <=> %s::vector 
                LIMIT 3;
            """
            cur.execute(search_query, (query_embedding,))
            relevant_docs = cur.fetchall()
        conn.close()

        if not relevant_docs:
            return "검색된 내용이 없습니다.", time.time() - start_time, False

        context = "\n\n".join([f"[{doc[1]} {doc[2]}페이지]: {doc[0]}" for doc in relevant_docs])

        system_instruction = (
            "당신은 제공된 논문 데이터를 기반으로 질문에 답하는 전문 어시스턴트입니다.\n"
            "다음 지침에 따라 답변을 생성하세요:\n\n"
            "1. 제공된 [논문 내용]에 질문과 관련된 정보가 조금이라도 있다면, 이를 활용해 최대한 친절하고 상세하게 답변하세요.\n"
            "2. 답변 시 반드시 해당 내용이 포함된 출처를 '[파일명 00페이지]' 형식으로 문장 속에 명시하세요.\n"
            "3. 만약 [논문 내용]이 질문과 전혀 상관이 없는 내용뿐이라면, 그때만 '제공된 논문에서 관련 정보를 찾을 수 없습니다'라고 답하세요.\n"
            "4. 논문 내용에 기반하되, 읽기 편하도록 논리적으로 요약하여 설명하세요."
        )

        responses = client.models.generate_content_stream(
            model="models/gemini-flash-latest",
            contents=f"{system_instruction}\n\n[논문 내용]:\n{context}\n\n질문: {user_query}"
        )
        
        return responses, start_time, False

    except Exception as e:
        st.error(f"오류 발생: {e}")
        return None, 0, False

# --- 4. 메인 UI 화면 ---
st.title("강원연구원 AI 테스트 [2026-03-15]")

# 기기 감지 및 안내 (기존 로직 유지)
ua_string = st.context.headers.get("User-Agent", "").lower()
if "android" in ua_string or "iphone" in ua_string:
    with st.expander("📱 홈 화면에 추가하여 앱처럼 사용하기"):
        st.write("메뉴에서 [홈 화면에 추가]를 누르면 바탕화면에서 바로 실행 가능합니다.")

# 채팅 기록 출력창
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(f'<div class="ai-answer">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.write(msg["content"])

# --- 5. [핵심] 입력창 처리 및 자동 비우기 로직 ---
# 사용자가 질문을 입력하고 화살표를 누르면 user_input에 값이 담깁니다.
user_input = st.chat_input("질문을 입력하세요 (강원)")

if user_input:
    # 1. 질문을 세션에 즉시 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    # 2. 강제 재실행 (이 시점에서 입력창의 글자가 싹 사라집니다)
    st.rerun()

# 3. 마지막 메시지가 'user'인 경우에만 AI 답변 생성 시작
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]
    
    with chat_container:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            result, start_t, is_cache = process_ai_query(last_query)
            
            if result:
                if is_cache:
                    full_response = result
                    placeholder.markdown(f'<div class="ai-answer">[Cache Hit]\n{full_response}</div>', unsafe_allow_html=True)
                else:
                    for chunk in result:
                        if chunk.text:
                            full_response += chunk.text
                            placeholder.markdown(f'<div class="ai-answer">{full_response}</div>', unsafe_allow_html=True)
                    
                    elapsed_time = time.time() - start_t
                    st.caption(f"총 소요 시간: {elapsed_time:.2f}초")
                
                # 답변이 완료되면 세션에 저장 (다음 rerun 때 화면에 유지됨)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.response_cache[last_query] = full_response
                # 답변 완료 후 깔끔하게 상태를 확정 짓기 위해 한 번 더 rerun (선택사항)
                st.rerun()
