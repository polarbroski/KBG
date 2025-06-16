from langchain_community.chat_models import ChatAnthropic
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def generate_summary_from_docs(player_name, documents):
    prompt_template = PromptTemplate.from_template(
        "Given the following documents, generate a scouting summary for {name}:\n\n{docs}"
    )

    prompt = prompt_template.format(name=player_name, docs='\n'.join(documents))

    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",  # 또는 최신 버전: "claude-3-5-sonnet-20240612"
        temperature=0.3,
        max_tokens=1024
    )

    return llm.predict(prompt)
