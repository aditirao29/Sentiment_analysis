from flask import Flask,render_template,request
import joblib
import re
import nltk
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.data.path.append(os.path.join(os.path.dirname(__file__),"nltk_data"))

# preprocessing
def to_lower(text):
    return text.lower()

def remove_special_characters(text):
    return re.sub(r'[^a-z0-9 ]','',text)

def tokenization(text):
    return word_tokenize(text)

def remove_punc(words):
    return [w.translate(str.maketrans('','',string.punctuation)) for w in words]
remove = remove_punc(stopwords.words('english'))

def remove_stopwords(words):
    return [w for w in words if w not in remove]

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemm = WordNetLemmatizer()

def lemmatization(words):
    pos_tags = nltk.pos_tag(words)
    return [lemm.lemmatize(w,get_pos(p)) for w,p in pos_tags]

def full_preprocess(text):
    text = to_lower(text)
    text = remove_special_characters(text)
    tokens = tokenization(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatization(tokens)
    return ' '.join(tokens)

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