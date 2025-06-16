from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def generate_summary_from_docs(player_name, documents):
    prompt_template = PromptTemplate.from_template(
        "Given the following documents, generate a scouting summary for {name}:\n\n{docs}"
    )
    prompt = prompt_template.format(name=player_name, docs='\n'.join(documents))

    llm = ChatOpenAI(temperature=0.3)
    return llm.predict(prompt)