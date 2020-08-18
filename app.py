import os
from flask import Flask,request, jsonify, render_template
import QA_model
model = QA_model.QuestionAnsweringModel()

app = Flask(__name__)
@app.route('/templates', methods=['GET','POST'])
def get_ans():
    if request.method == 'POST':
        Question = request.form['Question']
        Context = request.form['Context']
        
        output_summary = model.predict(Question, Context)
        
        return render_template('index.html', output_summary = output_summary)

@app.route('/')
def homepage():
    title = "TEXT summarizer"
    return render_template('index.html', title = title)

if __name__ == "__main__":
    app.run()