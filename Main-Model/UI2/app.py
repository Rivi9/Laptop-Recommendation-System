from flask import Flask, request, jsonify, render_template
import spacy
import pandas as pd
import fasttext
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load spaCy model for keyword extraction
nlp = spacy.load("en_core_web_lg")

# Load FastText model for sentence classification
model = fasttext.load_model("job_classification2.bin")

# Set of stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess the text using spaCy to extract keywords."""
    doc = nlp(text)
    price, ram, gpu = None, None, None
    words_between = []

    for i, token in enumerate(doc):
        if token.text.lower() == '$' and i + 1 < len(doc) and doc[i + 1].like_num:
            price = float(doc[i + 1].text)
        elif token.text.lower() == 'gb' and i - 1 >= 0 and doc[i - 1].like_num:
            ram = doc[i - 1].text + "GB"
        elif token.text.lower() in ['nvidia', 'amd', 'intel']:
            words_between.append(token.text)
            for j in range(i + 1, len(doc)):
                if doc[j].text.lower() in [".", "and", "gpu", ","]:
                    break
                else:
                    words_between.append(doc[j].text)
            gpu = ' '.join(words_between)
            break  # Assuming only one GPU mention per sentence

    return price, ram, gpu

def preprocess_with_stemming(text):
    """Preprocess text for classification with stemming."""
    text = re.sub(r'[^\w\s]', '', text).lower()
    text = re.sub(' +', ' ', text)
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def predict_category(sentence):
    """Predict the category using the FastText model."""
    preprocessed_sentence = preprocess_with_stemming(sentence)
    prediction = model.predict(preprocessed_sentence)
    return prediction[0][0], prediction[1][0]

@app.route('/')
def index():
    return render_template('sentenceUpdated.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentence = data['sentence']

    # Extract keywords first
    price, ram, gpu = preprocess_text(sentence)

    # If keywords are found, use them directly in the response
    if any([price, ram, gpu]):
        response = {
            'method': 'Keyword Extraction',
            'price': price,
            'ram': ram,
            'gpu': gpu
        }
    else:
        # Otherwise, use the classification model
        category, confidence = predict_category(sentence)
        response = {
            'method': 'Classification',
            'category': category,
            'confidence': confidence
        }

    return jsonify(response)

@app.route('/page1')
def page1():
    return render_template('page1Select.html')

if __name__ == '__main__':
    app.run(debug=True)
