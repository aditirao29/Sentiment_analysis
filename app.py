from flask import Flask,render_template,request
import joblib
import nltk
import os
from preprocessing import full_preprocess

nltk.data.path.append(os.path.join(os.path.dirname(__file__),"nltk_data"))

app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')

@app.route('/',methods=['GET','POST'])
def home():
    prediction = None
    if request.method=='POST':
        text = request.form['text']
        prediction = model.predict([text])[0]
    return render_template('index.html',prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)