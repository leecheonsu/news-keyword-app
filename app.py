ake_wordcloudimport streamlit as st
import requests
import pandas as pd
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta
import ast
from openai import OpenAI

# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="뉴스 키워드 분석기",
    page_icon="📰",
    layout="wide"
)

st.title("📰 뉴스 키워드 분석기")
st.write("검색어를 입력하면 최근 뉴스, 관련 키워드, 감성 분석, 기사 추세를 보여줍니다.")

NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# =========================
# 유틸 함수
# =========================
def clean_html(text):
    """HTML 태그 제거"""
    if not text:
        return ""

    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"&quot;|&apos;|&amp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_pub_date(pub_date):
    """네이버 뉴스 pubDate를 날짜 형식으로 변환"""
    try:
        dt = parsedate_to_datetime(pub_date)
        return dt.date()
    except Exception:
        return None


def extract_press_name(link):
    """
    간단한 언론사 추정.
    네이버 뉴스 API에는 언론사명이 직접 제공되지 않기 때문에
    링크 도메인 기준으로 임시 분류합니다.
    """
    if not link:
        return "알 수 없음"

    match = re.search(r"https?://([^/]+)", link)
    if match:
        domain = match.group(1)
        domain = domain.replace("www.", "")
        return domain

    return "알 수 없음"

def get_korean_press_name(domain):
    """
    기사 URL 도메인을 기준으로 한글 언론사명을 매핑합니다.
    매핑에 없는 도메인은 기존 도메인을 그대로 반환합니다.
    """

    press_map = {
        "4th.kr": "포쓰저널",
        "biz.heraldcorp.com": "헤럴드경제",
        "biz.newdaily.co.kr": "뉴데일리경제",
        "dailian.co.kr": "데일리안",
        "ebn.co.kr": "EBN",
        "econovill.com": "이코노믹리뷰",
        "enewstoday.co.kr": "이뉴스투데이",
        "hankyung.com": "한국경제",
        "ichannela.com": "채널A",
        "insight.co.kr": "인사이트",
        "jbnews.com": "중부매일",
        "joongangenews.com": "중앙이코노미뉴스",
        "mt.co.kr": "머니투데이",
        "news.bbsi.co.kr": "BBS뉴스",
        "newsis.com": "뉴시스",

        # 자주 나오는 주요 언론사 추가
        "yna.co.kr": "연합뉴스",
        "yonhapnewstv.co.kr": "연합뉴스TV",
        "chosun.com": "조선일보",
        "joongang.co.kr": "중앙일보",
        "donga.com": "동아일보",
        "hani.co.kr": "한겨레",
        "khan.co.kr": "경향신문",
        "mk.co.kr": "매일경제",
        "sedaily.com": "서울경제",
        "edaily.co.kr": "이데일리",
        "fnnews.com": "파이낸셜뉴스",
        "etnews.com": "전자신문",
        "zdnet.co.kr": "지디넷코리아",
        "bloter.net": "블로터",
        "thelec.kr": "디일렉",
        "ddaily.co.kr": "디지털데일리",
        "inews24.com": "아이뉴스24",
        "news1.kr": "뉴스1",
        "nocutnews.co.kr": "노컷뉴스",
        "ytn.co.kr": "YTN",
        "sbs.co.kr": "SBS",
        "kbs.co.kr": "KBS",
        "imbc.com": "MBC",
        "jtbc.co.kr": "JTBC"
    }

    if not domain:
        return "알 수 없음"

    domain = domain.replace("www.", "").strip()

    return press_map.get(domain, domain)

@st.cache_data(ttl=600)
def search_news(query, display=30):
    """네이버 뉴스 검색 API 호출"""
    url = "https://openapi.naver.com/v1/search/news.json"

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }

    params = {
        "query": query,
        "display": display,
        "start": 1,
        "sort": "date"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        st.error(f"API 오류가 발생했습니다. 상태 코드: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    items = data.get("items", [])

    rows = []
    for item in items:
        title = clean_html(item.get("title", ""))
        description = clean_html(item.get("description", ""))
        link = item.get("link", "")
        originallink = item.get("originallink", "")
        pub_date = item.get("pubDate", "")

        press_domain = extract_press_name(originallink or link)
        press_korean_name = get_korean_press_name(press_domain)

        rows.append({
            "title": title,
            "description": description,
            "link": link,
            "originallink": originallink,
            "pubDate": pub_date,
            "date": parse_pub_date(pub_date),
            "press": press_domain,
            "press_name": press_korean_name
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=86400)
def get_stock_code_from_krx(stock_name):
    """
    KRX 상장회사목록에서 종목명으로 국내 주식 종목코드를 찾습니다.
    예: 삼성전자 → 005930
    """

    if not stock_name:
        return None, None

    stock_name = stock_name.strip()

    # 사용자가 6자리 숫자 종목코드를 직접 넣은 경우
    if re.fullmatch(r"\d{6}", stock_name):
        return stock_name, stock_name

    try:
        url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"

        krx_df = pd.read_html(url, encoding="euc-kr")[0]

        krx_df["종목코드"] = krx_df["종목코드"].astype(str).str.zfill(6)

        # 완전 일치 우선
        exact_match = krx_df[krx_df["회사명"] == stock_name]

        if not exact_match.empty:
            row = exact_match.iloc[0]
            return row["종목코드"], row["회사명"]

        # 일부 일치 보조
        partial_match = krx_df[krx_df["회사명"].str.contains(stock_name, case=False, na=False)]

        if not partial_match.empty:
            row = partial_match.iloc[0]
            return row["종목코드"], row["회사명"]

        return None, None

    except Exception as e:
        st.warning(f"KRX 상장 종목 목록에서 종목코드를 찾지 못했습니다: {e}")
        return None, None
    

@st.cache_data(ttl=600)
def get_naver_stock_price_data(stock_code, period="3개월"):
    """
    네이버증권 일별 시세 데이터를 가져옵니다.
    국내 주식 기준입니다.
    선차트와 봉차트 모두 사용할 수 있도록
    시가, 고가, 저가, 종가, 거래량을 가져옵니다.
    """

    if not stock_code:
        return pd.DataFrame()

    period_days = {
        "1일": 5,
        "1주일": 10,
        "3개월": 100,
        "1년": 370,
        "3년": 365 * 3 + 30,
        "5년": 365 * 5 + 30,
        "10년": 365 * 10 + 30
    }

    days = period_days.get(period, 100)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    start_time = start_date.strftime("%Y%m%d")
    end_time = end_date.strftime("%Y%m%d")

    url = "https://api.finance.naver.com/siseJson.naver"

    params = {
        "symbol": stock_code,
        "requestType": "1",
        "startTime": start_time,
        "endTime": end_time,
        "timeframe": "day"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        text = response.text.strip()

        if not text or text == "[]":
            return pd.DataFrame()

        data = ast.literal_eval(text)

        if len(data) <= 1:
            return pd.DataFrame()

        columns = data[0]
        rows = data[1:]

        df_stock = pd.DataFrame(rows, columns=columns)

        df_stock["date"] = pd.to_datetime(
            df_stock["날짜"].astype(str),
            format="%Y%m%d"
        ).dt.date

        for col in ["시가", "고가", "저가", "종가", "거래량"]:
            df_stock[col] = pd.to_numeric(df_stock[col], errors="coerce")

        df_stock["일간 수익률"] = df_stock["종가"].pct_change() * 100

        df_stock = df_stock.dropna(subset=["종가"])

        return df_stock[
            ["date", "시가", "고가", "저가", "종가", "일간 수익률", "거래량"]
        ]

    except Exception as e:
        st.warning(f"네이버증권 주가 데이터를 가져오지 못했습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_data(ticker, period="1mo"):
    """
    yfinance를 이용해 주가 데이터를 가져옵니다.
    국내 종목 예시:
    삼성전자 005930.KS
    SK하이닉스 000660.KS
    에코프로비엠 247540.KQ

    미국 종목 예시:
    NVDA, TSLA, AAPL
    """
    if not ticker:
        return pd.DataFrame()

    try:
        stock = yf.Ticker(ticker)
        df_stock = stock.history(period=period)

        if df_stock.empty:
            return pd.DataFrame()

        df_stock = df_stock.reset_index()
        df_stock["date"] = pd.to_datetime(df_stock["Date"]).dt.date
        df_stock["종가"] = df_stock["Close"]
        df_stock["일간 수익률"] = df_stock["종가"].pct_change() * 100
        df_stock["거래량"] = df_stock["Volume"]

        return df_stock[["date", "종가", "일간 수익률", "거래량"]]

    except Exception as e:
        st.warning(f"주가 데이터를 가져오지 못했습니다: {e}")
        return pd.DataFrame()

def extract_keywords(text, custom_stopwords=None):
    """간단한 키워드 추출"""
    if custom_stopwords is None:
        custom_stopwords = set()

    basic_stopwords = {
        "뉴스", "기사", "기자", "관련", "사진", "지난", "이번", "대한",
        "있는", "없는", "한다", "했다", "위해", "통해", "최근", "오늘",
        "것으로", "이라고", "그리고", "하지만", "또한", "기준", "발표",
        "지난해", "올해", "내년", "이후", "전날", "오전", "오후",
        "머니투데이", "연합뉴스", "한국경제", "매일경제", "서울경제"
    }

    stopwords = basic_stopwords.union(custom_stopwords)

    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    words = text.split()

    words = [
        word for word in words
        if len(word) >= 2 and word not in stopwords
    ]

    return words


def get_korean_font_path():
    """
    로컬 Windows와 Streamlit Cloud Linux 환경에서
    각각 사용할 수 있는 한글 폰트 경로를 찾습니다.
    """

    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf",  # Windows 로컬
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Streamlit Cloud + fonts-nanum
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            return font_path

    return None


def make_wordcloud(words):
    """워드클라우드 생성"""
    text = " ".join(words)

    font_path = get_korean_font_path()

    if font_path is None:
        st.warning(
            "한글 폰트를 찾지 못했습니다. Streamlit Cloud에서는 packages.txt에 fonts-nanum을 추가해 주세요."
        )
        return None

    wc = WordCloud(
        font_path=font_path,
        width=900,
        height=500,
        background_color="white",
        max_words=100
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def analyze_sentiment(text):
    """
    키워드 기반 감성 분석.
    긍정/부정 키워드의 총 등장 횟수와 실제 등장한 키워드 목록을 함께 반환합니다.
    """
    positive_words = [
        "상승", "급등", "호조", "개선", "성장", "확대", "강세", "기대",
        "수혜", "흑자", "증가", "돌파", "최고", "회복", "긍정", "수주",
        "인상", "호실적", "신기록", "반등"
    ]

    negative_words = [
        "하락", "급락", "부진", "악화", "감소", "축소", "약세", "우려",
        "적자", "손실", "위기", "논란", "리스크", "부담", "침체",
        "경고", "불확실", "둔화", "폭락", "실패"
    ]

    positive_matches = {}
    negative_matches = {}

    for word in positive_words:
        count = text.count(word)
        if count > 0:
            positive_matches[word] = count

    for word in negative_words:
        count = text.count(word)
        if count > 0:
            negative_matches[word] = count

    pos_count = sum(positive_matches.values())
    neg_count = sum(negative_matches.values())

    if pos_count > neg_count:
        sentiment = "긍정"
    elif neg_count > pos_count:
        sentiment = "부정"
    else:
        sentiment = "중립"

    return sentiment, pos_count, neg_count, positive_matches, negative_matches


def generate_insight(keyword, keyword_df, sentiment, pos_count, neg_count, mode):
    """검색 결과 기반 인사이트 문장 생성"""
    top_words = keyword_df["단어"].head(5).tolist()

    if mode == "투자 분석용":
        insight = f"""
'{keyword}' 관련 최근 기사에서는 {", ".join(top_words)} 등의 단어가 자주 나타났습니다. 
전체적인 뉴스 분위기는 '{sentiment}'에 가깝게 나타났습니다. 
긍정 키워드는 {pos_count}회, 부정 키워드는 {neg_count}회 확인되었습니다. 
따라서 단순히 기사 수만 보기보다는 어떤 이슈가 반복적으로 언급되는지 확인할 필요가 있습니다. 
특히 투자 관점에서는 실적, 수주, 정책, 금리, 규제, 경쟁 환경과 연결되는 단어가 함께 등장하는지 추가로 살펴보는 것이 좋습니다.
"""
    elif mode == "평가 자료용":
        insight = f"""
'{keyword}' 관련 최근 기사에서는 {", ".join(top_words)} 등의 핵심어가 많이 나타났습니다. 
이를 통해 이 주제가 사회적으로 어떤 쟁점과 연결되어 있는지 파악할 수 있습니다. 
기사의 전체 분위기는 '{sentiment}'으로 나타났으며, 이는 해당 주제에 대한 사회적 평가나 우려를 분석하는 근거로 활용할 수 있습니다. 
수행평가에서는 단순히 기사 내용을 요약하기보다, 핵심 키워드를 바탕으로 원인, 영향, 해결 방안을 연결해 서술하면 좋습니다.
"""
    else:
        insight = f"""
'{keyword}' 관련 최근 기사에서는 {", ".join(top_words)} 등의 단어가 자주 등장했습니다. 
뉴스 분위기는 '{sentiment}'에 가깝게 나타났습니다. 
이 결과를 바탕으로 현재 해당 키워드가 어떤 이슈와 함께 다뤄지고 있는지 확인할 수 있습니다.
"""

    return insight.strip()


def make_simple_summary(df, keyword_df):
    """기사 요약문 생성"""
    top_words = keyword_df["단어"].head(5).tolist()

    summary = f"""
최근 검색된 기사들을 종합하면, 주요 키워드는 {", ".join(top_words)}입니다. 
기사들은 대체로 이 키워드들과 관련된 변화, 전망, 영향, 시장 반응을 다루고 있습니다. 
따라서 현재 이 이슈는 단순한 단발성 뉴스가 아니라 여러 기사에서 반복적으로 언급되는 관심 주제라고 볼 수 있습니다.
"""
    return summary.strip()


@st.cache_data(ttl=600)
def generate_ai_news_analysis(
    keyword,
    articles,
    sentiment,
    pos_count,
    neg_count,
    top_keywords,
    stock_name=None,
    stock_change_rate=None
):
    """
    OpenAI API를 이용해 기사 제목/요약을 자연스럽게 요약하고
    투자자용 핵심 이슈 3가지를 생성합니다.
    """

    if not OPENAI_API_KEY:
        return {
            "summary": "OpenAI API 키가 설정되어 있지 않습니다.",
            "issues": [],
            "investor_note": "secrets.toml에 OPENAI_API_KEY를 추가해 주세요."
        }

    # 너무 많은 기사를 그대로 보내면 비용이 커지므로 상위 20개만 사용
    article_texts = []
    for idx, row in articles.head(20).iterrows():
        article_texts.append(
            f"{idx + 1}. 제목: {row['title']}\n"
            f"   요약: {row['description']}\n"
            f"   출처: {row.get('press_name', row.get('press', ''))}\n"
        )

    article_block = "\n".join(article_texts)

    stock_context = ""
    if stock_name and stock_change_rate is not None:
        stock_context = f"""
주가 참고 정보:
- 종목명: {stock_name}
- 선택 기간 주가 변동률: {stock_change_rate:.2f}%
"""

    prompt = f"""
너는 한국 주식 시장을 분석하는 투자 리서치 보조 AI다.
아래 뉴스 검색 결과를 바탕으로 과장 없이, 투자자가 이해하기 쉽게 정리해라.

검색어: {keyword}

뉴스 분위기:
- 전체 분위기: {sentiment}
- 긍정 키워드 수: {pos_count}
- 부정 키워드 수: {neg_count}

자주 나온 키워드:
{", ".join(top_keywords)}

{stock_context}

기사 목록:
{article_block}

출력 형식은 반드시 아래 형식을 지켜라.

[AI 기사 종합 요약]
3~5문장으로 자연스럽게 요약한다.

[투자자용 핵심 이슈 3가지]
1. 이슈명: 설명
2. 이슈명: 설명
3. 이슈명: 설명

[투자 시 확인할 점]
투자자가 추가로 확인해야 할 점을 3가지 bullet로 작성한다.

주의:
- 기사에 없는 내용을 단정하지 마라.
- 매수/매도 추천을 하지 마라.
- 불확실한 내용은 '~가능성이 있다', '~확인할 필요가 있다'로 표현해라.
- 한국어로 작성해라.
"""

    try:
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=prompt
        )

        return {
            "summary": response.output_text,
            "issues": [],
            "investor_note": ""
        }

    except Exception as e:
        error_text = str(e)

        if "insufficient_quota" in error_text or "429" in error_text:
            return {
                "summary": (
                    "OpenAI API 사용 한도 또는 크레딧이 부족하여 AI 요약을 생성하지 못했습니다.\n\n"
                    "해결 방법:\n"
                    "- OpenAI Platform에서 Billing / Add credits를 확인하세요.\n"
                    "- API 사용 한도와 프로젝트 설정을 확인하세요.\n"
                    "- 크레딧 충전 후 Streamlit 앱을 다시 실행하세요.\n\n"
                    "현재는 기본 기사 요약과 자동 인사이트만 표시됩니다."
                ),
                "issues": [],
                "investor_note": "insufficient_quota"
            }

        return {
            "summary": f"AI 분석 중 오류가 발생했습니다: {e}",
            "issues": [],
            "investor_note": ""
        }

# =========================
# 사이드바 입력 영역
# =========================
st.sidebar.header("검색 설정")

def run_analysis():
    st.session_state["run_analysis"] = True


keyword = st.sidebar.text_input(
    "검색할 단어",
    placeholder="예: 인공지능, 삼성전자, 전기차, 반도체",
    on_change=run_analysis
)

article_count = st.sidebar.slider(
    "기사 개수",
    min_value=10,
    max_value=100,
    value=30,
    step=10
)

analysis_mode_options = {
    "일반 분석용": "(전체 흐름 파악용) 어떤 이슈와 연결되어 있는지를 중립적으로 정리",
    "투자 분석용": "주식·ETF·종목 이슈 분석용",
    "평가 자료용": "과제·보고서·발표 준비용"
}

analysis_mode = st.sidebar.selectbox(
    "인사이트 유형",
    list(analysis_mode_options.keys()),
    index=1,
    help=(
        "일반 분석용: 전체 흐름 파악용\n\n"
        "투자 분석용: 주식·ETF·종목 이슈 분석용\n\n"
        "평가 자료용: 과제·보고서·발표 준비용"
    )
)

st.sidebar.caption(f"💡 {analysis_mode_options[analysis_mode]}")

use_ai_analysis = st.sidebar.checkbox(
    "AI 기사 요약 사용",
    value=True,
    help="OpenAI API를 사용해 기사 요약과 투자자용 핵심 이슈 3가지를 생성합니다.",
    key="use_ai_analysis_checkbox"
)

stopword_input = st.sidebar.text_area(
    "제외할 단어 입력",
    placeholder="예: 기자, 사진, 단독, 속보",
    height=100
)

custom_stopwords = set(stopword_input.split())

# 뉴스 분석 버튼을 주가 연동 설정 위에 배치
analyze_button = st.sidebar.button(
    "뉴스 분석하기",
    key="news_analysis_button"
)

st.sidebar.divider()

st.sidebar.subheader("주가 연동 설정")

stock_name_input = st.sidebar.text_input(
    "종목명",
    placeholder="예: 삼성전자, SK하이닉스, 한화오션",
    key="stock_name_input"
)

stock_period = st.sidebar.selectbox(
    "주가 조회 기간",
    ["1일", "1주일", "3개월", "1년", "3년", "5년", "10년"],
    index=2,
    help="네이버증권 차트 기간과 비슷하게 조회합니다.",
    key="stock_period_selectbox"
)

stock_chart_type = st.sidebar.radio(
    "주가 차트 유형",
    ["선차트", "봉차트"],
    horizontal=True,
    key="stock_chart_type_radio"
)

st.sidebar.caption(
    "💡 종목명을 비워두면 위 검색어로 네이버증권 종목을 자동 조회합니다."
)


if analyze_button or st.session_state.get("run_analysis", False):
    st.session_state["run_analysis"] = False

    if not keyword:
        st.warning("검색어를 입력해 주세요.")
    else:
        with st.spinner("뉴스를 검색하고 분석하는 중입니다..."):
            df = search_news(keyword, article_count)

        if df.empty:
            st.warning("검색 결과가 없습니다.")
        else:
            # 전체 텍스트
            all_text = " ".join(df["title"].tolist() + df["description"].tolist())
            words = extract_keywords(all_text, custom_stopwords)
            counter = Counter(words)

            keyword_df = pd.DataFrame(
                counter.most_common(30),
                columns=["단어", "빈도"]
            )

            sentiment, pos_count, neg_count, positive_matches, negative_matches = analyze_sentiment(all_text)

            # =========================
            # 5차 기능: 네이버증권 주가 데이터 가져오기
            # =========================
            stock_search_name = stock_name_input.strip() if stock_name_input.strip() else keyword.strip()

            stock_code, resolved_stock_name = get_stock_code_from_krx(stock_search_name)

            if stock_code:
                stock_df = get_naver_stock_price_data(stock_code, stock_period)
            else:
                stock_df = pd.DataFrame()


            # =========================
            # 요약 지표
            # =========================
            st.subheader("분석 요약")

            def sentiment_color(value):
                if value == "긍정":
                    return "red"
                elif value == "부정":
                    return "royalblue"
                else:
                    return "#333333"


            news_mood_color = sentiment_color(sentiment)

            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

            with summary_col1:
                st.markdown(
                    f"""
                    <div style="padding: 10px 0;">
                        <div style="font-size: 15px; color: #333333;">검색 기사 수</div>
                        <div style="font-size: 36px; font-weight: 600; color: #222222;">{len(df)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with summary_col2:
                st.markdown(
                    f"""
                    <div style="padding: 10px 0;">
                        <div style="font-size: 15px; color: #333333;">뉴스 분위기</div>
                        <div style="font-size: 36px; font-weight: 600; color: {news_mood_color};">{sentiment}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with summary_col3:
                st.markdown(
                    f"""
                    <div style="padding: 10px 0;">
                        <div style="font-size: 15px; color: #333333;">긍정 키워드 수</div>
                        <div style="font-size: 36px; font-weight: 600; color: red;">{pos_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with summary_col4:
                st.markdown(
                    f"""
                    <div style="padding: 10px 0;">
                        <div style="font-size: 15px; color: #333333;">부정 키워드 수</div>
                        <div style="font-size: 36px; font-weight: 600; color: royalblue;">{neg_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =========================
            # 1차 기능: 기사 목록 + 워드클라우드
            # =========================
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "워드클라우드",
                "기사 목록",
                "추세 분석",
                "주가 비교",
                "인사이트"
            ])

            with tab1:
                st.subheader("워드클라우드와 관련 키워드")

                if words:
                    col_wc, col_kw = st.columns([2, 1])

                    with col_wc:
                        fig = make_wordcloud(words)

                        if fig is not None:
                            st.pyplot(fig)

                    with col_kw:
                        st.write("관련 키워드 TOP 30")
                        st.dataframe(keyword_df, use_container_width=True)
                else:
                    st.warning("분석할 단어가 부족합니다.")


            with tab2:
                title_col, download_col = st.columns([5, 1])

                with title_col:
                    st.subheader("최근 기사 목록")

                with download_col:
                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="CSV 다운로드",
                        data=csv,
                        file_name=f"{keyword}_news_data.csv",
                        mime="text/csv",
                        key="news_csv_download_button",
                        use_container_width=True
                    )

                for idx, row in df.iterrows():
                    st.markdown(f"### [{row['title']}]({row['link']})")
                    st.write(row["description"])
                    st.caption(f"날짜: {row['pubDate']} / 출처: {row['press_name']} ({row['press']})")
                    st.divider()

            # =========================
            # 2차 기능: 날짜별 기사 수, 언론사별 기사 수
            # =========================
            with tab3:
                st.subheader("날짜별 기사 수")

                date_df = (
                    df.dropna(subset=["date"])
                    .groupby("date")
                    .size()
                    .reset_index(name="기사 수")
                    .sort_values("date")
                )

                if not date_df.empty:
                    # 날짜를 보기 좋게 표시하기 위한 문자열 컬럼 생성
                    date_df["날짜"] = pd.to_datetime(date_df["date"]).dt.strftime("%m/%d")

                    fig_date = px.line(
                        date_df,
                        x="날짜",
                        y="기사 수",
                        markers=True,
                        title="날짜별 기사 수 추이"
                    )

                    fig_date.update_traces(
                        line=dict(width=4),
                        marker=dict(size=10),
                        hovertemplate=
                        "<b style='font-size:16px;'>날짜: %{x}</b><br>" +
                        "기사 수: %{y}건<extra></extra>"
                    )

                    fig_date.update_layout(
                        xaxis_title="날짜",
                        yaxis_title="기사 수",
                        xaxis=dict(
                            tickfont=dict(
                                size=18,
                                color="crimson"
                            ),
                            title_font=dict(
                                size=18,
                                color="crimson"
                            )
                        ),
                        yaxis=dict(
                            tickfont=dict(
                                size=13,
                                color="#333333"
                            ),
                            title_font=dict(
                                size=14,
                                color="#333333"
                            )
                        ),
                        title_font=dict(
                            size=22
                        ),
                        hoverlabel=dict(
                            font_size=15
                        )
                    )

                    st.plotly_chart(fig_date, use_container_width=True)

                else:
                    st.info("날짜 정보를 분석할 수 없습니다.")

                st.subheader("언론사별 기사 수")

                press_df = (
                    df.groupby(["press_name", "press"])
                    .size()
                    .reset_index(name="기사 수")
                    .sort_values("기사 수", ascending=False)
                    .head(15)
                )

                fig_press = px.bar(
                    press_df,
                    x="press_name",
                    y="기사 수",
                    text="기사 수",
                    title="언론사별 기사 수",
                    hover_data={
                        "press_name": True,
                        "press": True,
                        "기사 수": True
                    }
                )

                fig_press.update_traces(
                    textposition="outside",
                    hovertemplate=
                    "<b>%{x}</b><br>" +
                    "기사 수: %{y}건<br>" +
                    "도메인: %{customdata[0]}<extra></extra>"
                )

                fig_press.update_layout(
                    xaxis=dict(
                        title=dict(
                            text="언론사",
                            font=dict(size=16, color="#333333")
                        ),
                        tickfont=dict(size=13, color="#333333"),
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title=dict(
                            text="기사 수",
                            font=dict(size=16, color="#333333")
                        ),
                        tickfont=dict(size=13, color="#333333")
                    ),
                    title_font=dict(size=24),
                    hoverlabel=dict(font_size=14)
                )

                st.plotly_chart(fig_press, use_container_width=True)

                press_table_df = press_df.rename(
                    columns={
                        "press_name": "언론사명",
                        "press": "도메인"
                    }
                )

                st.dataframe(
                    press_table_df[["언론사명", "도메인", "기사 수"]],
                    use_container_width=True
                )

            # =========================
            # 5차 기능: 뉴스 기사 수와 네이버증권 주가 변동 비교
            # =========================
            with tab4:
                st.subheader("주가 데이터 연동")

                if not stock_search_name:
                    st.info("검색어 또는 종목명을 입력하면 네이버증권 주가 데이터와 뉴스 기사 수를 비교할 수 있습니다.")

                elif not stock_code:
                    st.warning(f"'{stock_search_name}'에 해당하는 국내 주식 종목을 네이버증권에서 찾지 못했습니다.")
                    st.caption("예: 삼성전자, SK하이닉스, 한화오션처럼 국내 주식 종목명을 입력해 주세요.")

                elif stock_df.empty:
                    st.warning("네이버증권 주가 데이터를 가져오지 못했습니다.")
                    st.caption(f"조회 종목: {resolved_stock_name} / 종목코드: {stock_code}")

                else:
                    st.caption(f"조회 종목: {resolved_stock_name} / 종목코드: {stock_code}")

                    latest_price = stock_df["종가"].iloc[-1]
                    first_price = stock_df["종가"].iloc[0]
                    price_change_rate = ((latest_price - first_price) / first_price) * 100
                    latest_volume = stock_df["거래량"].iloc[-1]

                    stock_col1, stock_col2, stock_col3 = st.columns(3)

                    with stock_col1:
                        st.markdown("#### 최근 종가")
                        st.markdown(f"## {latest_price:,.0f}원")

                    with stock_col2:
                        color = "red" if price_change_rate >= 0 else "royalblue"
                        st.markdown("#### 기간 수익률")
                        st.markdown(
                            f"<h2 style='color:{color};'>{price_change_rate:.2f}%</h2>",
                            unsafe_allow_html=True
                        )

                    with stock_col3:
                        st.markdown("#### 최근 거래량")
                        st.markdown(f"## {latest_volume:,.0f}")

                    st.divider()

                    st.subheader("주가 추이")

                    stock_chart_df = stock_df.copy()

                    # 조회 기간이 길면 날짜 표기를 연/월 중심으로, 짧으면 월/일 중심으로 표시
                    if stock_period in ["3년", "5년", "10년"]:
                        stock_chart_df["날짜"] = pd.to_datetime(stock_chart_df["date"]).dt.strftime("%Y-%m")
                    else:
                        stock_chart_df["날짜"] = pd.to_datetime(stock_chart_df["date"]).dt.strftime("%m/%d")

                    if stock_chart_type == "선차트":
                        fig_price = px.line(
                            stock_chart_df,
                            x="날짜",
                            y="종가",
                            markers=True,
                            title=f"{resolved_stock_name} 주가 추이 - {stock_period} / 선차트"
                        )

                        fig_price.update_traces(
                            line=dict(width=4),
                            marker=dict(size=7),
                            hovertemplate=
                            "<b>날짜: %{x}</b><br>" +
                            "종가: %{y:,.0f}원<extra></extra>"
                        )

                    else:
                        fig_price = go.Figure(
                            data=[
                                go.Candlestick(
                                    x=stock_chart_df["날짜"],
                                    open=stock_chart_df["시가"],
                                    high=stock_chart_df["고가"],
                                    low=stock_chart_df["저가"],
                                    close=stock_chart_df["종가"],
                                    increasing_line_color="red",
                                    decreasing_line_color="royalblue",
                                    name="봉차트"
                                )
                            ]
                        )

                        fig_price.update_layout(
                            title=f"{resolved_stock_name} 주가 추이 - {stock_period} / 봉차트",
                            xaxis_rangeslider_visible=False
                        )

                    fig_price.update_layout(
                        xaxis=dict(
                            title=dict(
                                text="날짜",
                                font=dict(size=16, color="#333333")
                            ),
                            tickfont=dict(size=13, color="#333333")
                        ),
                        yaxis=dict(
                            title=dict(
                                text="주가",
                                font=dict(size=16, color="#333333")
                            ),
                            tickfont=dict(size=13, color="#333333")
                        ),
                        title_font=dict(size=22),
                        hoverlabel=dict(font_size=14)
                    )

                    st.plotly_chart(fig_price, use_container_width=True)

                    st.subheader("뉴스 기사 수와 주가 변동 비교")

                    st.info(
                        "네이버 뉴스 검색은 최근 기사 중심으로 수집되기 때문에, "
                        "기사 수는 특정 날짜에 몰릴 수 있습니다. "
                        "따라서 이 그래프는 뉴스가 실제로 검색된 날짜 기준으로 해석하는 것이 좋습니다."
                    )

                    # 날짜별 기사 수 데이터 생성
                    news_count_df = (
                        df.dropna(subset=["date"])
                        .groupby("date")
                        .size()
                        .reset_index(name="기사 수")
                        .sort_values("date")
                    )

                    if news_count_df.empty:
                        st.info("뉴스 날짜 데이터가 부족해 주가와 비교할 수 없습니다.")

                    else:
                        news_start_date = news_count_df["date"].min()
                        news_end_date = news_count_df["date"].max()

                        st.caption(
                            f"뉴스 데이터 기준 기간: {news_start_date} ~ {news_end_date} "
                            f"/ 검색된 기사 {len(df)}건 기준"
                        )

                        # 뉴스가 나온 날짜와 주가 데이터를 병합
                        compare_df = pd.merge(
                            news_count_df,
                            stock_df,
                            on="date",
                            how="left"
                        )

                        compare_df["날짜"] = pd.to_datetime(compare_df["date"]).dt.strftime("%m/%d")
                        compare_df["일간 수익률"] = compare_df["일간 수익률"].fillna(0)

                        # 뉴스 날짜가 너무 적은 경우 안내
                        if len(compare_df) <= 2:
                            st.warning(
                                "검색된 뉴스가 특정 날짜에 집중되어 있어, "
                                "기사 수와 주가 수익률의 관계를 해석하기에는 데이터가 부족할 수 있습니다."
                            )

                        fig_compare = go.Figure()

                        # 뉴스 기사 수 막대
                        fig_compare.add_trace(
                            go.Bar(
                                x=compare_df["날짜"],
                                y=compare_df["기사 수"],
                                name="뉴스 기사 수",
                                text=compare_df["기사 수"],
                                textposition="outside",
                                marker_color="#4C78A8",
                                yaxis="y",
                                hovertemplate=
                                "<b>%{x}</b><br>" +
                                "뉴스 기사 수: %{y}건<extra></extra>"
                            )
                        )

                        # 일간 수익률 선
                        fig_compare.add_trace(
                            go.Scatter(
                                x=compare_df["날짜"],
                                y=compare_df["일간 수익률"],
                                name="일간 수익률(%)",
                                mode="lines+markers",
                                marker=dict(size=9, color="crimson"),
                                line=dict(width=3, color="crimson"),
                                yaxis="y2",
                                hovertemplate=
                                "<b>%{x}</b><br>" +
                                "일간 수익률: %{y:.2f}%<extra></extra>"
                            )
                        )

                        fig_compare.update_layout(
                            title="뉴스 발생일 기준 기사 수와 주가 수익률 비교",
                            xaxis=dict(
                                title=dict(
                                    text="날짜",
                                    font=dict(size=16, color="#333333")
                                ),
                                tickfont=dict(size=14, color="#333333")
                            ),
                            yaxis=dict(
                                title=dict(
                                    text="뉴스 기사 수",
                                    font=dict(size=15, color="#333333")
                                ),
                                tickfont=dict(size=13, color="#333333"),
                                rangemode="tozero"
                            ),
                            yaxis2=dict(
                                title=dict(
                                    text="일간 수익률(%)",
                                    font=dict(size=15, color="crimson")
                                ),
                                tickfont=dict(size=13, color="crimson"),
                                overlaying="y",
                                side="right",
                                zeroline=True,
                                zerolinecolor="crimson"
                            ),
                            title_font=dict(size=22),
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )

                        st.plotly_chart(
                            fig_date,
                            use_container_width=True,
                            key="date_news_count_chart"
                        )

                        st.caption(
                            "이 그래프는 전체 주가 기간이 아니라, 뉴스가 실제로 검색된 날짜를 기준으로 비교합니다."
                        )

                        # 표도 함께 보여주기
                        compare_table_df = compare_df[["date", "기사 수", "종가", "일간 수익률"]].copy()
                        compare_table_df = compare_table_df.rename(
                            columns={
                                "date": "날짜",
                                "종가": "종가",
                                "일간 수익률": "일간 수익률(%)"
                            }
                        )

                        compare_table_df["일간 수익률(%)"] = compare_table_df["일간 수익률(%)"].round(2)

                        st.dataframe(
                            compare_table_df,
                            use_container_width=True
                        )

                    st.plotly_chart(
                        fig_press,
                        use_container_width=True,
                        key="press_count_chart"
                    )

                    st.caption("막대는 뉴스가 검색된 날짜의 기사 수, 선 그래프는 해당 날짜의 주가 일간 수익률입니다.")

                    valid_corr_df = compare_df.dropna(subset=["일간 수익률"])

                    if len(valid_corr_df) >= 2:
                        correlation = valid_corr_df["기사 수"].corr(valid_corr_df["일간 수익률"])

                        st.subheader("뉴스 기사 수와 주가 변동의 단순 상관관계")

                        if pd.isna(correlation):
                            st.info("상관관계를 계산하기에는 데이터가 부족합니다.")
                        else:
                            st.metric("상관계수", f"{correlation:.2f}")

                            if correlation > 0.3:
                                st.success("기사 수가 많은 날에 주가 수익률도 함께 높아지는 경향이 일부 나타났습니다.")
                            elif correlation < -0.3:
                                st.warning("기사 수가 많은 날에 주가 수익률이 낮아지는 경향이 일부 나타났습니다.")
                            else:
                                st.info("기사 수와 주가 수익률 사이의 뚜렷한 관계는 약하게 나타났습니다.")

                    st.divider()

                    st.subheader("종목 뉴스 대시보드 요약")

                    dashboard_summary = f"""
                    {resolved_stock_name} 관련 검색 결과를 보면, 최근 기사 수는 총 {len(df)}건입니다. 
                    뉴스 분위기는 '{sentiment}'으로 나타났고, 긍정 키워드는 {pos_count}회, 부정 키워드는 {neg_count}회 감지되었습니다. 
                    선택한 기간 동안 주가는 {price_change_rate:.2f}% 변동했습니다. 
                    따라서 이 종목은 뉴스 키워드, 기사 수 변화, 주가 흐름을 함께 보면서 해석하는 것이 좋습니다.
                    """

                    st.info(dashboard_summary.strip())

            # =========================
            # 3차 기능: 감성 분석 + 인사이트 + 요약
            # =========================
            with tab5:
                st.subheader("간단 감성 분석")

                positive_keyword_text = ", ".join(
                    [f"{word}({count}회)" for word, count in positive_matches.items()]
                )

                negative_keyword_text = ", ".join(
                    [f"{word}({count}회)" for word, count in negative_matches.items()]
                )

                if not positive_keyword_text:
                    positive_keyword_text = "감지된 긍정 키워드 없음"

                if not negative_keyword_text:
                    negative_keyword_text = "감지된 부정 키워드 없음"

                sentiment_df = pd.DataFrame({
                    "구분": ["긍정 키워드", "부정 키워드"],
                    "횟수": [pos_count, neg_count],
                    "감지된 키워드": [positive_keyword_text, negative_keyword_text],
                    "색상": ["긍정", "부정"]
                })

                fig_sentiment = px.bar(
                    sentiment_df,
                    x="구분",
                    y="횟수",
                    color="색상",
                    color_discrete_map={
                        "긍정": "red",
                        "부정": "royalblue"
                    },
                    text="횟수",
                    hover_data={
                        "구분": True,
                        "횟수": True,
                        "감지된 키워드": True,
                        "색상": False
                    },
                    title="긍정/부정 키워드 감성 분석"
                )

                fig_sentiment.update_traces(
                    textposition="outside",
                    hovertemplate=
                    "<b>%{x}</b><br>" +
                    "총 등장 횟수: %{y}회<br>" +
                    "키워드: %{customdata[0]}<extra></extra>"
                )

                fig_sentiment.update_layout(
                    xaxis_title="감성 구분",
                    yaxis_title="키워드 등장 횟수",
                    showlegend=False
                )

                st.plotly_chart(fig_sentiment, use_container_width=True)

                st.subheader("AI 기사 요약 및 투자 이슈")

                top_keywords = keyword_df["단어"].head(10).tolist()

                # 주가 변동률이 있으면 AI 분석에 같이 전달
                ai_stock_name = None
                ai_stock_change_rate = None

                try:
                    if "resolved_stock_name" in locals() and "price_change_rate" in locals():
                        ai_stock_name = resolved_stock_name
                        ai_stock_change_rate = price_change_rate
                except Exception:
                    pass

                if use_ai_analysis:
                    with st.spinner("AI가 기사 내용을 요약하고 투자 이슈를 정리하는 중입니다..."):
                        ai_result = generate_ai_news_analysis(
                            keyword=keyword,
                            articles=df,
                            sentiment=sentiment,
                            pos_count=pos_count,
                            neg_count=neg_count,
                            top_keywords=top_keywords,
                            stock_name=ai_stock_name,
                            stock_change_rate=ai_stock_change_rate
                        )

                    if ai_result.get("investor_note") == "insufficient_quota":
                        st.warning(ai_result["summary"])

                        st.subheader("기본 기사 요약")
                        summary = make_simple_summary(df, keyword_df)
                        st.info(summary)

                    else:
                        st.subheader("AI 기사 종합 요약")
                        st.markdown(ai_result["summary"])

                else:
                    st.subheader("기본 기사 요약")
                    summary = make_simple_summary(df, keyword_df)
                    st.info(summary)

                st.subheader("자동 인사이트")

                insight = generate_insight(
                    keyword,
                    keyword_df,
                    sentiment,
                    pos_count,
                    neg_count,
                    analysis_mode
                )

                st.success(insight)

else:
    st.info("왼쪽 사이드바에서 검색어를 입력하고 '뉴스 분석하기' 버튼을 눌러주세요.")