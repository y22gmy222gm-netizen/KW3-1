#
# streamlit run test1.py
# https://www.youtube.com/watch?v=6_0_kV08HX8
# 

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
    /* 1. 제목 크기 반응형 (이전보다 조금 더 작게 조절) */
    h1 {
        font-size: clamp(1.1rem, 4.5vw, 2.2rem) !important;
        word-break: keep-all;
    }

    /* 2. 답변 글자 스타일 */
    .ai-answer { color: blue; white-space: pre-wrap; font-size: 11pt; }
    
    /* 3. 상단 헤더 및 기본 메뉴 숨기기 */
    header, footer, #MainMenu { visibility: hidden !important; }

    /* 4. [중요] 채팅 입력창 위치를 왕관 아이콘 위로 강제 이동 */
    /* 아이콘에 가려지지 않도록 바닥에서 80px 이상 띄웁니다. */
    div[data-testid="stChatInput"] {
        bottom: 90px !important; 
        z-index: 100 !important;
        background-color: rgba(255, 255, 255, 0.9) !important; /* 약간 투명한 흰색 배경 */
        border-radius: 15px !important;
        padding: 5px !important;
    }

    /* 5. 본문 내용이 입력창에 가려지지 않게 하단 여백 확보 */
    .main .block-container {
        padding-bottom: 180px !important;
    }
    
    /* 6. 모바일에서 전송 버튼이 잘 보이도록 설정 */
    [data-testid="stChatInputSubmit"] {
        color: white !important;
        background-color: #FF4B4B !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 초기 설정 및 클라이언트 (보안 수정 완료) ---
# 실제 값은 Streamlit Cloud의 Settings -> Secrets에 넣으세요.
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

# --- [추가] 기기 감지 및 홈 화면 추가 안내 로직 ---
ua_string = st.context.headers.get("User-Agent", "").lower()

if "android" in ua_string:
    with st.expander("📱 안드로이드: 이 앱을 홈 화면에 추가하여 사용하세요"):
        st.write("1. 브라우저 우측 상단 **점 3개(⋮)** 메뉴를 누릅니다.")
        st.write("2. **[홈 화면에 추가]** 또는 **[앱 설치]**를 선택하세요.")
        st.write("3. 이제 바탕화면 아이콘을 통해 앱처럼 바로 접속 가능합니다!")
elif "iphone" in ua_string or "ipad" in ua_string:
    with st.expander("🍎 아이폰: 이 앱을 홈 화면에 추가하여 사용하세요"):
        st.write("1. 브라우저 하단 중앙의 **공유 버튼(□↑)**을 누릅니다.")
        st.write("2. 메뉴를 아래로 내려 **[홈 화면에 추가]**를 선택하세요.")
        st.write("3. 바탕화면에 아이콘이 생성되어 앱처럼 쓸 수 있습니다!")
else:
    st.caption("💡 PC에서 접속 중입니다. 모바일에서도 동일한 주소로 사용 가능합니다.")

# 채팅 기록 출력창
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(f'<div class="ai-answer">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.write(msg["content"])

# 입력창
if user_input := st.chat_input("질문을 입력하세요 (강원)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)

    with chat_container:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            result, start_t, is_cache = process_ai_query(user_input)
            
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
                
                st.session_state.response_cache[user_input] = full_response
                st.session_state.messages.append({"role": "assistant", "content": full_response})
