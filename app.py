from flask import Flask, render_template, request
from generate_report import generate_scouting_report
from retriever import retrieve_documents
from rag_llm import generate_summary_from_docs

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form['name']
        avg = float(request.form['avg'])
        hr = int(request.form['hr'])
        run_time = float(request.form['run_time'])
        is_lhh = request.form['is_lhh'] == 'True'
        pop_time = float(request.form['pop_time']) if request.form['pop_time'] else None

        player_info = {
            'name': name,
            'avg': avg,
            'hr': hr,
            'run_time': run_time,
            'is_lhh': is_lhh,
            'pop_time': pop_time
        }

        docs = retrieve_documents(name)
        llm_summary = generate_summary_from_docs(name, docs)
        report = generate_scouting_report(player_info)

        return render_template("result.html", name=name, summary=llm_summary, report=report)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
