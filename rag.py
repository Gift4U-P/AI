import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

vectorstore = None
retriever = None
llm = None

def initialize_rag():
    global vectorstore, retriever, llm
    
    possible_files = ['present_dataset.xlsx - naver_gift_recommendation_datas.csv', 'present_dataset.csv', 'present_dataset.xlsx']
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if not file_path:
        print("❌ 데이터를 찾을 수 없습니다.")
        return

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
        df = df.fillna('')
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    documents = []
    for _, row in df.iterrows():
        text_content = f"상품: {row.get('title','')} | 가격: {row.get('lprice','')} | 카테고리: {row.get('category1','')} {row.get('category2','')} {row.get('category3','')}"
        metadata = {
            "title": str(row.get('title', '')),
            "lprice": str(row.get('lprice', '')),
            "link": str(row.get('link', '')),
            "image": str(row.get('image', '')),
            "mallName": str(row.get('mallName', '네이버쇼핑'))
        }
        documents.append(Document(page_content=text_content, metadata=metadata))

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️ OPENAI_API_KEY가 없습니다.")
        return

    print("📥 임베딩 생성 중... (OpenAI text-embedding-3-small)")
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        print("✅ 벡터 DB 구축 완료!")
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}")
        return
    
    print("☁️ LLM 연결 중...")
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_key
        )
        print("✅ LLM 연결 성공!")
    except Exception as e:
        print(f"❌ OpenAI 연결 실패: {e}")
        llm = None

    print("✅ RAG 시스템 초기화 완료")

# --- Helper Functions ---
def convert_keywords_to_query(kw: dict) -> str:
    return f"선물 받는 사람은 {kw.get('age', '')}세 {kw.get('gender', '')}이며, 관계는 '{kw.get('relationship', '')}'입니다. 현재 '{kw.get('situation', '')}' 상황에 맞는 선물을 찾고 있습니다."

def convert_survey_to_query(ans: dict) -> str:
    traits = []
    def get_score(q1_key, q2_key):
        score = 0
        if ans.get(q1_key) == 'A': score += 1
        if ans.get(q2_key) == 'A': score += 1
        return score

    score_e = get_score('q1', 'q2')
    if score_e == 2: traits.append("사교적이고 활동적이며 사람들과의 만남과 새로운 경험을 즐기는 편입니다")
    elif score_e == 1: traits.append("상황에 따라 사람들과의 만남이나 외부 활동을 즐기는 편입니다")
    else: traits.append("차분하고 조용한 환경을 선호하며 실내 활동을 편안하게 느끼는 편입니다")

    score_a = get_score('q3', 'q4')
    if score_a == 2: traits.append("타인의 감정에 민감하고 갈등을 부드럽게 해결하는 배려심 많은 성향입니다")
    elif score_a == 1: traits.append("상황에 따라 배려심을 보이지만 갈등 해결 방식은 중간 정도입니다")
    else: traits.append("타인의 감정보다 자신의 기준을 우선시하는 경향이 있습니다")

    score_c = get_score('q5', 'q6')
    if score_c == 2: traits.append("계획적이고 실용적인 것을 선호하며, 꼼꼼하고 책임감 있는 성향입니다")
    elif score_c == 1: traits.append("실용성과 감성을 적절히 균형 있게 고려하는 성향입니다")
    else: traits.append("계획적이기보다는 즉흥적이고 감성적인 요소를 더 중요하게 느끼는 성향입니다")

    score_n = get_score('q7', 'q8')
    if score_n == 2: traits.append("스트레스에 취약한 편이며, 안정감과 편안함을 주는 것을 선호합니다")
    elif score_n == 1: traits.append("상황에 따라 스트레스를 받을 수 있으며 어느 정도 안정감을 필요로 합니다")
    else: traits.append("감정 기복이 적고 스트레스 상황에서도 비교적 안정적인 편입니다")

    score_o = get_score('q9', 'q10')
    if score_o == 2: traits.append("새로운 활동이나 독창적인 아이디어에 열려 있으며 감각적이고 창의적인 스타일을 좋아합니다")
    elif score_o == 1: traits.append("새로운 경험에도 열려 있으면서 익숙한 스타일도 선호하는 균형 잡힌 성향입니다")
    else: traits.append("익숙하고 전통적인 스타일을 선호하며 변화보다는 안정감을 편안하게 느낍니다")

    return f"이 사람은 {', '.join(traits)}. 이 사람에게 가장 잘 어울리는 선물을 추천해줘."

# --- [Function 1] 설문조사 추천 함수 (기존 로직) ---
def get_survey_recommendation(user_query: str):
    if not retriever or not llm:
        return None

    searched_docs = retriever.invoke(user_query)
    product_context = "\n".join([f"- {d.page_content}" for d in searched_docs])
    
    template = """
    당신은 선물 추천 전문가 'Gift4U'입니다.
    
    [사용자 입력 정보]
    "{user_query}"
    
    [검색된 상품 후보]
    {product_context}

    위 정보를 바탕으로 한국어로 답변해주세요.
    반드시 아래 형식(ANALYSIS, REASONING, MESSAGE)을 지켜주세요.
    
    [출력 형식]:
    ANALYSIS: (성향 및 상황 분석)
    REASONING: (추천 이유)
    MESSAGE: (카드 메시지)
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        response_text = chain.invoke({"user_query": user_query, "product_context": product_context})
        analysis, reasoning, message = parse_llm_response(response_text)
        
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        analysis, reasoning, message = "분석 오류", "추천 오류", "메시지 생성 오류"

    gift_list = []
    for doc in searched_docs:
        gift_list.append({
            "title": doc.metadata.get("title", ""),
            "link": doc.metadata.get("link", ""),
            "image": doc.metadata.get("image", ""),
            "lprice": doc.metadata.get("lprice", ""),
            "mallName": doc.metadata.get("mallName", "")
        })

    return {
        "analysis": analysis,
        "reasoning": reasoning,
        "card_message": message,
        "giftList": gift_list
    }

# --- [Function 2] 키워드 추천 함수 (신규 로직) ---
def get_keyword_recommendation(keyword_dict: dict):
    if not retriever or not llm:
        return None

    # 1. 검색 쿼리 변환
    user_query = convert_keywords_to_query(keyword_dict)
    
    # 2. 유사도 검색
    searched_docs = retriever.invoke(user_query)
    product_context = "\n".join([f"- {d.page_content}" for d in searched_docs])
    
    # 3. keywordText 생성 (규칙 기반)
    # 예: "20대 연인 1주년 기념일에는 이런 선물을 추천해요"
    age = keyword_dict.get('age', '')
    relation = keyword_dict.get('relationship', '')
    situation = keyword_dict.get('situation', '')
    keyword_text = f"{age} {relation} {situation}에는 이런 선물을 추천해요"

    # 4. LLM 메시지 생성 (오직 메시지만 생성)
    template = """
    당신은 선물 추천 전문가 'Gift4U'입니다.
    
    사용자 정보: "{user_query}"
    추천된 상품들: {product_context}

    위 정보를 바탕으로 선물과 함께 보낼 센스있고 감동적인 [카드 메시지] 하나만 작성해주세요.
    다른 설명 없이 메시지 내용만 바로 출력하세요.
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        card_message = chain.invoke({"user_query": user_query, "product_context": product_context})
        card_message = card_message.strip().replace('"', '') # 따옴표 제거 등 정제
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        card_message = "행복한 하루 되세요!"

    # 5. 선물 리스트 매핑
    gift_list = []
    for doc in searched_docs:
        gift_list.append({
            "title": doc.metadata.get("title", ""),
            "link": doc.metadata.get("link", ""),
            "image": doc.metadata.get("image", ""),
            "lprice": doc.metadata.get("lprice", ""),
            "mallName": doc.metadata.get("mallName", "")
        })

    # 6. 최종 결과 반환 (명세서 형식에 맞춤)
    return {
        "age": keyword_dict.get('age'),
        "gender": keyword_dict.get('gender'),
        "relationship": keyword_dict.get('relationship'),
        "situation": keyword_dict.get('situation'),
        "keywordText": keyword_text,
        "card_message": card_message,
        "giftList": gift_list
    }

def parse_llm_response(text):
    analysis = ""
    reasoning = ""
    message = ""
    current_section = None
    for line in text.split('\n'):
        line = line.strip()
        if "ANALYSIS:" in line:
            current_section = "analysis"
            analysis += line.replace("ANALYSIS:", "").strip() + " "
        elif "REASONING:" in line:
            current_section = "reasoning"
            reasoning += line.replace("REASONING:", "").strip() + " "
        elif "MESSAGE:" in line:
            current_section = "message"
            message += line.replace("MESSAGE:", "").strip() + " "
        elif current_section == "analysis":
            analysis += line + " "
        elif current_section == "reasoning":
            reasoning += line + " "
        elif current_section == "message":
            message += line + " "
    return analysis.strip(), reasoning.strip(), message.strip()