# app.py
from flask import Flask, render_template, request
from rag_llm import generate_summary_from_docs
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        batting_avg = request.form["batting_avg"]
        home_runs = request.form["home_runs"]
        run_time = request.form["run_time"]
        left_handed = request.form["left_handed"]
        pop_time = request.form.get("pop_time", "")

        summary_filename = generate_summary_from_docs(name, [])
        return render_template("result.html", report_img=summary_filename)

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
