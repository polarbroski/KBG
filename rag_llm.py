from langchain_community.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate

def generate_summary_from_docs(player_name, documents):
    # 프롬프트 템플릿 정의
    prompt_template = PromptTemplate.from_template(
        "Given the following documents, generate a scouting summary for {name}:\n\n{docs}"
    )

    # 플레이어 이름과 문서들을 프롬프트에 삽입
    prompt = prompt_template.format(
        name=player_name,
        docs='\n'.join(documents)
    )

    # Claude 모델 초기화 (2024년 6월 기준 최신 Sonnet 버전 권장)
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",  # 또는 최신: "claude-3-5-sonnet-20240612"
        temperature=0.3,
        max_tokens=1024
    )

    # 예측 실행
    return llm.predict(prompt)
