from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def generate_summary_from_docs(player_name, documents):
    prompt_template = PromptTemplate.from_template(
        "Given the following documents, generate a scouting summary for {name}:\n\n{docs}"
    )

    prompt = prompt_template.format(name=player_name, docs='\n'.join(documents))

    # 모델명을 명시적으로 설정 (접근 가능한 모델명으로)
    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-0125")  # 또는 "gpt-4", "gpt-4o"

    return llm.predict(prompt)
