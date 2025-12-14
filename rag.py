import os
import pandas as pd
import random
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
global_df = None


# --- Trait Keyword Mapping (Step 1: foundation for tag-based re-ranking) ---
# Big5 설문 결과(높음/보통/낮음)를 "비교 가능한 키워드"로 변환하기 위한 테이블입니다.
# Step 2에서 상품의 personality_tags와 교집합 점수(trait match score)를 계산할 때 사용됩니다.
TRAIT_KEYWORDS = {
    "외향성": {
        "높음": ["활동적", "사교적", "야외", "경험", "여행", "모임"],
        "보통": ["취미", "균형", "상황에따라"],
        "낮음": ["조용한", "실내", "혼자", "집콕", "힐링", "편안한"]
    },
    "우호성": {
        "높음": ["배려", "따뜻한", "감성", "의미있는", "선물용"],
        "보통": ["무난한", "실용적"],
        "낮음": ["개성", "자기취향", "독립적", "독특한"]
    },
    "성실성": {
        "높음": ["실용적", "기능성", "정리", "계획적", "꼼꼼한"],
        "보통": ["일상용", "무난한"],
        "낮음": ["즉흥", "디자인", "감성", "재미"]
    },
    "신경성": {
        "높음": ["안정", "힐링", "편안한", "스트레스완화", "휴식"],
        "보통": ["무난한", "일상용"],
        "낮음": ["도전", "활력", "자극", "야외"]
    },
    "개방성": {
        "높음": ["트렌디", "창의적", "새로운", "취미", "감각적"],
        "보통": ["세련된", "적당한", "무난한"],
        "낮음": ["클래식", "전통", "익숙한", "안정적인"]
    }
}

def initialize_rag():
    global vectorstore, retriever, llm, global_df
    
    possible_files = ['present_dataset.xlsx - naver_gift_recommendation_datas.csv', 'present_dataset.csv', 'present_dataset.xlsx']
    file_path = next((f for f in possible_files if os.path.exists(f)), None)
    
    if not file_path:
        print("데이터를 찾을 수 없습니다.")
        return

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
        df = df.fillna('')
        
        def clean_price(x):
            try:
                return int(str(x).replace(',', '').replace('원', '').strip())
            except:
                return 0
        
        df['lprice_int'] = df['lprice'].apply(clean_price)
        global_df = df

    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return

    documents = []

    for _, row in df.iterrows():
        title = str(row.get('title', ''))
        categories = [
            str(row.get('category1', '')),
            str(row.get('category2', '')),
            str(row.get('category3', '')),
        ]
        try:
            lprice_int = int(row.get('lprice_int', 0))
        except Exception:
            lprice_int = 0

        # Step 2: product personality tags (dataset column if present; else inferred)
        raw_tags = row.get('personality_tags', '')
        product_tags = _parse_tag_field(raw_tags)
        if not product_tags:
            product_tags = infer_product_personality_tags(title=title, categories=categories, lprice_int=lprice_int)

        # Step 6: infer target gender for filtering (F/M/U)
        target_gender = infer_product_target_gender(title=title, categories=categories)
        target_gender_text = "여성" if target_gender == "F" else ("남성" if target_gender == "M" else "")

        tags_text = ", ".join(product_tags) if product_tags else ""

        text_content = (
            f"상품: {title} | 가격: {row.get('lprice','')} | "
            f"카테고리: {row.get('category1','')} {row.get('category2','')} {row.get('category3','')}"
        )
        # Enrich the searchable text for better retrieval
        if tags_text:
            text_content += f" | 성향태그: {tags_text}"
        if target_gender_text:
            text_content += f" | 추천성별: {target_gender_text}"

        metadata = {
            "title": title,
            "lprice": str(row.get('lprice', '')),
            "link": str(row.get('link', '')),
            "image": str(row.get('image', '')),
            "mallName": str(row.get('mallName', '네이버쇼핑')),
            "category1": str(row.get('category1', '')),
            "category2": str(row.get('category2', '')),
            "category3": str(row.get('category3', '')),
            "target_gender": target_gender,
            "personality_tags": product_tags
        }
        documents.append(Document(page_content=text_content, metadata=metadata))

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY가 없습니다.")
        return

    print("임베딩 생성 중... (OpenAI text-embedding-3-small)")
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key
        )
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # 참고: 실제 검색에서는 vectorstore를 직접 사용하여 점수를 가져옵니다.
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6, 
                "fetch_k": 30,
                "lambda_mult": 0.6
            }
        )
        print("벡터 DB 및 검색 시스템 구축 완료!")
    except Exception as e:
        print(f"검색 엔진 생성 실패: {e}")
        return
    
    print("LLM 연결 중...")
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_key
        )
        print("LLM 연결 성공!")
    except Exception as e:
        print(f"OpenAI 연결 실패: {e}")
        llm = None

    print("RAG 시스템 초기화 완료")


# --- Product Tagging Rules (Step 2: foundation for tag-based matching) ---
# Goal: ensure each product has "personality_tags" even if the dataset does not include them yet.
# You can later replace these heuristics by adding a `personality_tags` column in the dataset.

# --- Product Tagging Rules (dataset-aware, no dataset edit required) ---

# Canonicalize similar tags into a smaller shared vocabulary that aligns with TRAIT_KEYWORDS.
TAG_CANONICAL_MAP = {
    "스타일": "세련된",
    "프리미엄": "세련된",
    "고급": "세련된",
    "가성비": "실용적",
    "일상용": "실용적",
    "자기관리": "세련된",
    "재미": "취미",
    "힐링템": "힐링",
    "스트레스 완화": "스트레스완화",
    "전통적인": "전통",
    "클래식한": "클래식",
}

# Category substring mapping (works for joined category1~4 text)
CATEGORY_TAGS = {
    # dataset category1
    "패션잡화": ["세련된", "개성", "트렌디"],
    "패션의류": ["세련된", "트렌디"],
    "가구/인테리어": ["감성", "힐링", "실내"],
    "생활/건강": ["실용적", "기능성", "익숙한"],
    "식품": ["무난한", "배려", "익숙한"],
    "출산/육아": ["의미있는", "따뜻한", "배려"],

    # dataset category2
    "주얼리": ["세련된", "감성", "의미있는", "클래식"],
    "여성가방": ["실용적", "세련된", "무난한"],
    "남성가방": ["실용적", "세련된", "무난한"],
    "지갑": ["실용적", "클래식", "무난한"],
    "문구/사무용품": ["실용적", "계획적", "정리", "창의적"],
    "인테리어소품": ["감성", "힐링", "실내"],
    "홈데코": ["감성", "힐링", "실내"],
    "주방용품": ["실용적", "정리", "익숙한"],
    "수납/정리용품": ["정리", "계획적", "실용적"],
    "안마용품": ["힐링", "안정", "스트레스완화", "기능성"],
    "건강식품": ["기능성", "배려", "익숙한"],
    "음료": ["감성", "무난한", "사교적"],
    "주류": ["경험", "사교적", "감성"],
    "원예/식물": ["감성", "힐링", "실내"],
    "출산/돌기념품": ["의미있는", "따뜻한", "배려"],
}

# Exact category (category3/4) mapping – uses the dataset's most frequent category3 values
CATEGORY_EXACT_TAGS = {
    "토트백": ["실용적", "세련된", "무난한"],
    "백팩": ["활동적", "실용적", "여행"],
    "책가방": ["실용적", "계획적"],
    "가방": ["실용적", "세련된", "무난한"],
    "남성지갑": ["실용적", "클래식", "무난한"],
    "카드/명함지갑": ["실용적", "클래식", "무난한"],

    "목걸이": ["세련된", "감성", "의미있는"],
    "반지": ["세련된", "감성", "의미있는"],
    "시계": ["클래식", "세련된", "실용적"],

    "필기도구": ["실용적", "계획적", "정리"],
    "문구용품": ["실용적", "계획적", "정리"],
    "카드/엽서/봉투": ["감성", "의미있는"],

    "잔/컵": ["감성", "무난한", "익숙한"],
    "식기": ["실용적", "정리", "익숙한"],
    "정리함": ["정리", "계획적", "실용적"],
    "거울": ["세련된", "실용적"],

    "아로마/캔들용품": ["힐링", "감성", "안정", "실내"],
    "오르골": ["감성", "힐링", "의미있는", "실내"],
    "비누꽃": ["감성", "따뜻한", "배려", "의미있는"],

    "차류": ["힐링", "감성", "안정"],
    "영양제": ["기능성", "배려", "익숙한"],
    "홍삼": ["기능성", "배려", "익숙한"],

    "이벤트/파티용품": ["사교적", "경험", "활동적"],
}

# Title keyword rules – tuned to your dataset titles (e.g., 집들이/각인/만년필/디퓨저 등)
TITLE_KEYWORD_TAGS = [
    (["각인", "주문제작", "제작", "커스텀", "이니셜", "포토", "사진"], ["의미있는", "개성", "감성"]),
    (["커플", "연인", "여자친구", "남자친구"], ["감성", "의미있는", "세련된"]),
    (["부모님"], ["배려", "익숙한", "기능성"]),
    (["집들이", "신혼부부"], ["실내", "익숙한", "의미있는"]),
    (["졸업", "입학", "합격", "학사모"], ["의미있는", "계획적"]),
    (["돌", "백일", "출산"], ["의미있는", "따뜻한", "배려"]),

    (["디퓨저", "캔들", "아로마", "향", "무드등"], ["힐링", "감성", "안정", "실내"]),
    (["마사지", "안마", "찜질", "족욕"], ["힐링", "안정", "스트레스완화", "기능성"]),
    (["만년필", "필기도구", "문구세트"], ["세련된", "실용적", "계획적"]),
    (["다이어리", "플래너", "노트"], ["계획적", "정리", "실용적"]),

    (["토트백", "백팩", "가방", "지갑"], ["실용적", "무난한", "세련된"]),
    (["14k", "18k", "골드", "실버", "은", "진주", "다이아"], ["세련된", "감성", "의미있는", "클래식"]),
    (["가죽"], ["클래식", "세련된", "실용적"]),

    (["홍삼", "영양제", "건강"], ["기능성", "배려", "익숙한"]),
    (["차", "티", "허브"], ["힐링", "감성", "안정"]),
    (["전통주", "막걸리", "청주"], ["전통", "경험", "사교적"]),
    (["유기", "전통", "고블렛", "사케잔"], ["전통", "클래식", "익숙한"]),

    (["선물세트", "세트"], ["선물용", "무난한"]),
]


def _parse_tag_field(value) -> list:
    """Parse dataset tag field into a list of strings.
    Accepts: list, 'a|b|c', 'a,b,c', 'a / b', etc.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    # normalize delimiters
    for delim in ["|", ",", "/", ";", "，"]:
        s = s.replace(delim, "|")
    tags = [t.strip() for t in s.split("|") if t.strip()]
    # de-dup while keeping order
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    # Keep tag set compact (helps matching + explainability)
    return out[:6]


def infer_product_target_gender(title: str, categories: list):
    """Infer target gender for a product: 'F' (female), 'M' (male), 'U' (unisex/unknown).

    We avoid using overly-broad keywords (e.g., '타이') to prevent false positives like '타이거', '타이머'.
    """
    title = str(title or "")
    categories = categories or []
    cat_text = " ".join([str(c or "") for c in categories])

    # Category-based strong signals
    if "여성" in cat_text:
        return "F"
    if "남성" in cat_text:
        return "M"

    t = title.lower()

    # Female markers
    female_markers = [
        "여성", "여자", "여친", "여자친구", "여성용", "여성가방", "여성지갑",
        "립스틱", "틴트", "쿠션", "파운데이션", "블러셔", "아이섀도", "마스카라",
        "네일", "향수", "핸드크림", "바디미스트", "스킨케어", "화장품"
    ]

    # Male markers
    male_markers = [
        "남성", "남자", "남친", "남자친구", "남성용", "남성가방", "남성지갑",
        "넥타이", "타이핀", "커프스", "면도", "쉐이빙", "정장", "벨트", "포마드"
    ]

    if any(m in t for m in female_markers):
        return "F"
    if any(m in t for m in male_markers):
        return "M"

    return "U"

def infer_product_personality_tags(title: str, categories: list, lprice_int: int = 0) -> list:
    """Heuristic tagging for products when dataset does not provide personality_tags.

    This is intentionally rule-based so you can use it WITHOUT editing the dataset file.
    It uses:
      - category1~4 (substring + exact mapping)
      - title keywords (dataset-tuned)
      - optional price hints (kept minimal)
    """
    tags = []

    # Clean categories
    cats = []
    for c in (categories or []):
        s = str(c).strip()
        if s and s.lower() != "nan":
            cats.append(s)

    # Category substring mapping (category1~4 joined)
    joined_cats = " ".join(cats)
    for key, tlist in CATEGORY_TAGS.items():
        if key and key in joined_cats:
            tags.extend(tlist)

    # Category exact mapping (mainly category3/4)
    for c in cats:
        if c in CATEGORY_EXACT_TAGS:
            tags.extend(CATEGORY_EXACT_TAGS[c])

    # Title keyword mapping
    t = (title or "").strip()
    for keywords, tlist in TITLE_KEYWORD_TAGS:
        if any(k in t for k in keywords):
            tags.extend(tlist)

    # Price hints (optional, keep inside canonical vocabulary)
    if isinstance(lprice_int, int) and lprice_int >= 200000:
        tags.extend(["세련된", "클래식"])
    elif isinstance(lprice_int, int) and 0 < lprice_int <= 50000:
        tags.extend(["실용적", "무난한"])

    # Canonicalize + de-dup while keeping order
    seen = set()
    out = []
    for tag in tags:
        tag = str(tag).strip()
        if not tag:
            continue
        tag = TAG_CANONICAL_MAP.get(tag, tag)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    # Keep tag set compact (helps matching + explainability)
    return out[:6]


# --- Helper Functions ---
def _normalize_tokens(tokens):
    """Normalize tokens for matching.

    - trim
    - lowercase
    - remove spaces
    - apply TAG_CANONICAL_MAP
    """
    if not tokens:
        return []
    out = []
    for t in tokens:
        s = str(t).strip().lower()
        if not s:
            continue
        # remove internal spaces for robust matching (e.g., '스트레스 완화' vs '스트레스완화')
        s = s.replace(" ", "")
        s = TAG_CANONICAL_MAP.get(s, s)
        out.append(s)
    return out


def trait_match_score(user_keywords, product_tags):
    """Return 0~1 trait match score based on overlap between user keywords and product tags.

    NOTE: We use 'precision' (how much of the product's tags are covered by user keywords),
    because user keywords are often longer than product tags (and recall-based scoring
    makes scores look artificially low).
    """
    if not user_keywords or not product_tags:
        return 0.0
    u = set(_normalize_tokens(user_keywords))
    p = set(_normalize_tokens(product_tags))
    if not u or not p:
        return 0.0
    matched = u & p
    return len(matched) / max(len(p), 1)

def combined_match_score(vector_score, trait_score, w_vector=0.6, w_trait=0.4):
    """Combine vector relevance score and trait score into a final 0~1 match score."""
    try:
        vs = float(vector_score)
    except Exception:
        vs = 0.0
    try:
        ts = float(trait_score)
    except Exception:
        ts = 0.0
    vs = max(0.0, min(1.0, vs))
    ts = max(0.0, min(1.0, ts))
    return (w_vector * vs) + (w_trait * ts)

def convert_keywords_to_query(kw: dict) -> str:
    return f"선물 받는 사람은 {kw.get('age', '')}세 {kw.get('gender', '')}이며, 관계는 '{kw.get('relationship', '')}'입니다. 현재 '{kw.get('situation', '')}' 상황에 맞는 선물을 찾고 있습니다."

def convert_survey_to_query(ans: dict):
    traits = []
    evidence_list = []
    user_trait_keywords = []

    def process_trait(q1_key, q2_key, trait_key, category_name, descriptions):
        valid_answers = []
        val1 = str(ans.get(q1_key, '')).upper()
        val2 = str(ans.get(q2_key, '')).upper()

        if val1 != 'C' and val1 in ['A', 'B']:
            valid_answers.append(val1)
        if val2 != 'C' and val2 in ['A', 'B']:
            valid_answers.append(val2)

        if not valid_answers:
            return

        a_count = valid_answers.count('A')
        ratio = a_count / len(valid_answers)

        description = ""
        trait_summary = ""
        level_label = ""

        if ratio == 1.0:
            description = descriptions[0]
            level_label = "높음"
            trait_summary = descriptions[3] + " (높음)"
        elif ratio == 0.5:
            description = descriptions[1]
            level_label = "보통"
            trait_summary = descriptions[3] + " (보통)"
        else:
            description = descriptions[2]
            level_label = "낮음"
            trait_summary = descriptions[3] + " (낮음)"

        evidence_list.append({
            "category": category_name,
            "description": description
        })
        traits.append(trait_summary)

        # Step 1: 설문 결과를 키워드로 변환해 누적 (Step 2에서 상품 태그와 비교 예정)
        kw_list = TRAIT_KEYWORDS.get(trait_key, {}).get(level_label, [])
        user_trait_keywords.extend(kw_list)

    process_trait('q1', 'q2', "외향성", "외향성 (Extraversion)", (
        "Q1, Q2에서 새로운 사람들과의 교류를 즐기고 외부 활동을 선호한다고 응답하여, 경험 중심 활동형 선물이 잘 맞는 성향으로 나타났습니다.",
        "Q1, Q2에서 상황에 따라 외부 활동을 즐기기도 하지만 혼자만의 시간도 중요시하는 균형 잡힌 성향으로 확인되었습니다.",
        "Q1, Q2에서 조용한 환경과 실내 활동을 선호한다고 응답하여, 집에서 즐길 수 있는 편안한 선물이 적합합니다.",
        "활동적이고 사교적인 성향"
    ))

    process_trait('q3', 'q4', "우호성", "우호성 (Agreeableness)", (
        "Q3, Q4 응답을 기반으로 타인의 감정을 세심하게 배려하고 관계를 중시하는 따뜻한 성향이 확인되었습니다.",
        "Q3, Q4 응답을 기반으로 타인의 분위기나 감정을 적당히 고려하지만, 필요할 때는 의견을 명확히 표현하는 균형 잡힌 대인 관계 스타일을 보였습니다.",
        "Q3, Q4 응답에서 타인의 시선보다는 자신의 주관과 기준을 뚜렷하게 가지고 있는 독립적인 성향이 나타났습니다.",
        "타인을 배려하는 우호적인 성향"
    ))

    process_trait('q5', 'q6', "성실성", "성실성 (Conscientiousness)", (
        "Q5, Q6 응답에서 계획적이고 실용적인 것을 선호하며, 물건을 고를 때도 기능성을 최우선으로 고려하는 꼼꼼한 성향이 확인되었습니다.",
        "Q5, Q6 응답에서 일정 관리와 실용성에 대해 과도하게 엄격하지도 느슨하지도 않은 보통 수준의 성향이 확인되었습니다.",
        "Q5, Q6 응답에서 계획보다는 즉흥적인 즐거움을 추구하고, 실용성보다는 디자인과 감성을 중요시하는 성향으로 나타났습니다.",
        "계획적이고 성실한 성향"
    ))

    process_trait('q7', 'q8', "신경성", "신경성 (Neuroticism)", (
        "Q7, Q8에서 변화나 돌발 상황에 민감하고 감정 기복이 어느 정도 있다는 응답을 바탕으로, 안정·힐링 중심의 선물이 특히 도움이 될 가능성이 높습니다.",
        "Q7, Q8 응답을 통해 일상적인 스트레스를 받기도 하지만, 적절히 대처하며 안정을 찾는 보통 수준의 정서 상태가 확인되었습니다.",
        "Q7, Q8 응답에서 스트레스 상황에서도 침착함을 유지하며 감정 기복이 적은 안정적인 성향이 확인되었습니다.",
        "정서적 안정이 필요한 성향"
    ))

    process_trait('q9', 'q10', "개방성", "개방성 (Openness)", (
        "Q9, Q10에서 새로운 취미, 창의적인 아이템, 감각적인 디자인을 선호한다는 선택이 확인되어, 개성 있는 취향 기반 선물을 선호하는 성향으로 판단됩니다.",
        "Q9, Q10 응답에서 익숙한 것과 새로운 것 사이에서 적절한 균형을 유지하며, 너무 튀지 않으면서도 세련된 스타일을 선호합니다.",
        "Q9, Q10 응답에서 유행을 타지 않는 클래식하고 전통적인 스타일을 선호하며, 익숙한 것에서 편안함을 느끼는 성향입니다.",
        "새로운 경험과 개방적인 성향"
    ))

    # 키워드 중복 제거 (순서 유지)
    deduped_keywords = []
    seen = set()
    for kw in user_trait_keywords:
        if kw and kw not in seen:
            deduped_keywords.append(kw)
            seen.add(kw)
    user_trait_keywords = deduped_keywords

    if not traits:
        query_str = "무난하고 인기 있는 선물을 추천해줘."
    else:
        query_str = f"이 사람은 {', '.join(traits)}입니다. 이 사람에게 가장 잘 어울리는 선물을 추천해줘."

    # Step 1 반환: (검색용 쿼리, 설명용 evidence, 비교용 trait keywords)
    return query_str, evidence_list, user_trait_keywords

# --- Recommendation Functions ---


def get_survey_recommendation(user_query: str, evidence_list: list, user_trait_keywords: list = None):
    if not vectorstore or not llm:
        return None

    # 1) Retrieve a broader candidate set first (for reranking)
    candidate_k = 12
    top_k = 6

    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(user_query, k=candidate_k)

    # 2) Rerank with trait keyword overlap
    scored_candidates = []
    for doc, score in docs_and_scores:
        vector_score = float(score)
        product_tags = doc.metadata.get("personality_tags", []) or []
        t_score = trait_match_score(user_trait_keywords or [], product_tags)
        final_score = combined_match_score(vector_score, t_score, w_vector=0.6, w_trait=0.4)
        scored_candidates.append((doc, final_score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_results = scored_candidates[:top_k]

    # LLM context: use only top results (already reranked)
    searched_docs = [doc for doc, _ in top_results]
    product_context = "\n".join([f"- {d.page_content}" for d in searched_docs])

    # Also pass user_trait_keywords for better explanations (LLM should not change ranking)
    user_trait_keywords_str = ", ".join(user_trait_keywords) if user_trait_keywords else ""

    template = """
    당신은 선물 추천 전문가 'Gift4U'입니다.
    사용자 정보: "{user_query}"
    사용자 성향 키워드: {user_trait_keywords_str}

    추천 상품 목록(이미 선정됨):
    {product_context}

    위 정보를 바탕으로 한국어로 답변해주세요.
    센스있고 감동적인 카드 메시지를 작성해주세요.(따옴표 없이)
    반드시 아래 형식(ANALYSIS, REASONING, MESSAGE)을 지켜주세요.

    - 추천 순서/상품 선택을 바꾸지 마세요. (이미 선정된 상품 목록을 설명만 하세요.)
    - 상품에 없는 기능/특징은 절대 만들어내지 마세요. 제공된 정보만 사용하세요.

    [출력 형식]:
    ANALYSIS: (성향 및 상황 분석)
    REASONING: (추천 이유)
    MESSAGE: (카드 메시지)
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    try:
        response_text = chain.invoke({
            "user_query": user_query,
            "product_context": product_context,
            "user_trait_keywords_str": user_trait_keywords_str
        })
        analysis, reasoning, message = parse_llm_response(response_text)
    except Exception as e:
        print(f"OpenAI Error: {e}")
        analysis, reasoning, message = "분석 오류", "추천 오류", "메시지 생성 오류"

    gift_list = []

    # Normalize scores relative to the best result so that the top item is ~1.0.
    # This prevents "scores looking too low" due to conservative relevance-score scaling.
    best_raw = max((raw for _, raw in top_results), default=0.0)
    if best_raw <= 0:
        best_raw = 1.0

    for doc, raw_final in top_results:
        normalized = raw_final / best_raw
        sim_score = round(float(normalized), 2)

        gift_list.append({
            "title": doc.metadata.get("title", ""),
            "link": doc.metadata.get("link", ""),
            "image": doc.metadata.get("image", ""),
            "lprice": doc.metadata.get("lprice", ""),
            "mallName": doc.metadata.get("mallName", ""),
            # 기존 응답 스키마 유지: accuracy 필드를 '상대 매칭 점수(최고=1.0)'로 사용
            "accuracy": sim_score
        })


    return {
        "analysis": analysis,
        "evidence": evidence_list,
        "reasoning": reasoning,
        "card_message": message,
        "giftList": gift_list
    }


def get_keyword_recommendation(keyword_dict: dict):
    if not vectorstore or not llm:
        return None

    user_query = convert_keywords_to_query(keyword_dict)

    # Step 6: widen candidates then apply gender filter (F/M/U)
    candidate_k = 40
    top_k = 6

    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(user_query, k=candidate_k)

    # Normalize user gender input
    gender_raw = str(keyword_dict.get('gender', '') or '').strip().lower()
    desired_gender = None
    if gender_raw:
        if "여" in gender_raw:
            desired_gender = "F"
        elif "남" in gender_raw:
            desired_gender = "M"

    if desired_gender:
        filtered = []
        for doc, score in docs_and_scores:
            tg = doc.metadata.get("target_gender")
            if not tg:
                # fallback inference if metadata is missing
                title = doc.metadata.get("title", "")
                cats = [doc.metadata.get("category1", ""), doc.metadata.get("category2", ""), doc.metadata.get("category3", "")]
                tg = infer_product_target_gender(title=title, categories=cats)

            # Allow unisex/unknown items as well
            if tg in (desired_gender, "U"):
                filtered.append((doc, score))

        # If we have enough items after filtering, use them; otherwise fall back to original
        if len(filtered) >= 3:
            docs_and_scores = filtered

    # Take top_k after filtering (docs_and_scores is already sorted by relevance in practice)
    docs_and_scores = docs_and_scores[:top_k]

    searched_docs = [doc for doc, _ in docs_and_scores]
    product_context = "\n".join([f"- {d.page_content}" for d in searched_docs])

    age = keyword_dict.get('age', '')
    relation = keyword_dict.get('relationship', '')
    situation = keyword_dict.get('situation', '')
    keyword_text = f"{age} {relation} {situation}에는 이런 선물을 추천해요"

    template = """
    당신은 선물 추천 전문가 'Gift4U'입니다.
    사용자 정보: "{user_query}"
    추천된 상품들: {product_context}

    위 정보를 바탕으로 선물과 함께 보낼 센스있고 감동적인 [카드 메시지] 하나만 작성해주세요.
    다른 설명 없이 메시지 내용만 바로 출력하세요. (따옴표 없이)
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    try:
        card_message = chain.invoke({"user_query": user_query, "product_context": product_context})
        card_message = card_message.strip().replace('"', '')
    except Exception as e:
        print(f"OpenAI Error: {e}")
        card_message = "행복한 하루 되세요!"

    gift_list = []

    # Normalize scores relative to the best result so that the top item is ~1.0.
    best_raw = max((float(score) for _, score in docs_and_scores), default=0.0)
    if best_raw <= 0:
        best_raw = 1.0

    for doc, score in docs_and_scores:
        normalized = float(score) / best_raw
        sim_score = round(normalized, 2)

        gift_list.append({
            "title": doc.metadata.get("title", ""),
            "link": doc.metadata.get("link", ""),
            "image": doc.metadata.get("image", ""),
            "lprice": doc.metadata.get("lprice", ""),
            "mallName": doc.metadata.get("mallName", ""),
            "accuracy": sim_score
        })

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

# --- 메인 홈 랜덤 추천 함수 ---
def get_main_page_gifts():
    if global_df is None:
        return None

    def df_to_gift_list(dataframe, count=6):
        sample_n = min(len(dataframe), count)
        if sample_n == 0: return []
        
        sampled = dataframe.sample(n=sample_n)
        result = []
        for _, row in sampled.iterrows():
            result.append({
                "title": str(row.get('title', '')),
                "link": str(row.get('link', '')),
                "image": str(row.get('image', '')),
                "lprice": str(row.get('lprice', '')),
                "mallName": str(row.get('mallName', '네이버쇼핑')),
                "accuracy": None
            })
        return result

    random_gifts = df_to_gift_list(global_df)
    luxury_df = global_df[global_df['lprice_int'] >= 200000]
    luxury_gifts = df_to_gift_list(luxury_df)
    budget_df = global_df[global_df['lprice_int'] <= 50000]
    budget_gifts = df_to_gift_list(budget_df)

    return {
        "randomGifts": random_gifts,
        "luxuryGifts": luxury_gifts,
        "budgetGifts": budget_gifts
    }