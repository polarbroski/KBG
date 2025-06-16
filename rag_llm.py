from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

def generate_summary_from_docs(player_name, documents):
    prompt_template = PromptTemplate.from_template(
        "Given the following documents, generate a scouting summary for {name}:\n\n{docs}"
    )

    prompt = prompt_template.format(
        name=player_name,
        docs='\n'.join(documents)
    )

    # ✅ langchain-anthropic에서 제공하는 ChatAnthropic 사용
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0.3,
        max_tokens=1024
    )

    return llm.invoke(prompt)
