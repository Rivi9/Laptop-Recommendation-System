from flask import Flask, render_template, request, jsonify
import pandas as pd
import difflib
import spacy
import fasttext
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

app = Flask(__name__)
laptops = pd.read_csv("LaptopsNew.csv")

# Mapping between predicted categories and dataset use cases
category_mapping = {
    "student_education": "student_education",
    "gaming": "gaming",
    "basic": "basic",
    "it": "it",
    "business_professional": "business_professional",
    "creative_design": "creative_design"
}

# price range mappings
price_range_mappings = {
    "Less than $500": (None, 500),
    "$500 - $1000": (500, 1000),
    "$1000 - $1500": (1000, 1500),
    "More than $1500": (1500, None),
}


# Preprocessing function
def preprocess_data():
    laptops.dropna(inplace=True)
    
    laptops['processor'] = laptops['processor'].apply(lambda x: x.split())
    laptops['os'] = laptops['os'].apply(lambda x: x.split())
    laptops['ram'] = laptops['ram'].astype(str)  # Converting to string for concatenation
    laptops['use'] = laptops['usecases'].apply(lambda x: x.split() if isinstance(x, str) else [])
    laptops['tags'] = laptops.apply(lambda x: x['processor'] + [x['ram']] + x['os'] + x['use'], axis=1)
    
    new = laptops.drop(columns=['processor', 'ram', 'os', 'storage', 'rating', 'os_brand', 'processor_brand', 'use'])
    new['tags'] = new['tags'].apply(lambda x: " ".join(x)).apply(lambda x: x.lower())
    
    ps = PorterStemmer()
    new['tags'] = new['tags'].apply(lambda x: " ".join([ps.stem(i) for i in x.split()]))
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(new['tags']).toarray()
    
    similarity = cosine_similarity(vector)
    
    return new, similarity

new, similarity = preprocess_data()

# Recommendation function
def recommend(use):
    mapped_use = category_mapping.get(use, use)  # Apply category mapping

    try:
        index = new[new['usecases'].str.lower() == mapped_use.lower()].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_names = [new.iloc[i[0]]['name'] for i in distances[1:6]]
        return recommended_names
    except IndexError:
        return ["No recommendations found for this use case"]


# Load spaCy model for keyword extraction
nlp = spacy.load("en_core_web_lg")

# Load FastText model for sentence classification
model = fasttext.load_model("input_classificationNew.bin")

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
    # Remove the "__label__" prefix and convert the category to lowercase
    category = prediction[0][0].replace("__label__", "").lower()
    confidence = prediction[1][0]
    return category, confidence

def filter_laptops(price=None, ram=None, gpu=None, priceRange=None):
    filtered_df = laptops.copy()
    
    if priceRange in price_range_mappings:
        min_price, max_price = price_range_mappings[priceRange]
        if min_price is not None:
            filtered_df = filtered_df[filtered_df['price'] >= min_price]
        if max_price is not None:
            filtered_df = filtered_df[filtered_df['price'] <= max_price]


    if price is not None:
        filtered_df = filtered_df[filtered_df['price'] <= price]

    if ram is not None:
        ram_value = int(ram.replace("GB", ""))  # Convert input to integer
        filtered_df = filtered_df[filtered_df['ram'].astype(int) >= ram_value]
    
    if gpu is not None:
        filtered_df = filtered_df[filtered_df['processor'].apply(lambda x: gpu.lower() in " ".join(x).lower())]
    
    return filtered_df



def recommend_based_on_filter(filtered_df):
    # Sort by price or any other preferred metric
    sorted_df = filtered_df.sort_values(by='price', ascending=True)
    # Select the top 5 recommendations
    recommendations = sorted_df.head(5)['name'].tolist()
    return recommendations


@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('page1Select.html')

@app.route('/select_dropdown', methods=['POST'])
def select_dropdown():
    return render_template('recommendationPage2.html')

@app.route('/select_sentence', methods=['POST'])
def select_sentence():
    return render_template('sentence.html')

@app.route('/page1')
def page1():
    return render_template('page1Select.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentence = data['sentence']
    priceRange = data.get('priceRange', None)  # Extract price range from the request


    # Extract keywords first
    price, ram, gpu = preprocess_text(sentence)
    recommendations = []  # Initialize recommendations list

    # Attempt to use extracted keywords or category for recommendations
    if any([price, ram, gpu]):
        # If specific keywords were found, use them for recommendations
        filtered_df = filter_laptops(price=price, ram=ram, gpu=gpu, priceRange=priceRange)
        # Proceed to generate recommendations based on filtered_df
        # This part needs to be adapted based on how you want to recommend based on filtered data
        recommendations = recommend_based_on_filter(filtered_df)

       
    else:
        # No specific keywords, use the classification model
        category, confidence = predict_category(sentence)
        filtered_df = filter_laptops(priceRange=priceRange)  # Filter based on price range
        temp_recommendations = recommend(category)
        # Filter the recommendations further based on the price range
        filtered_recommendations = filtered_df[filtered_df['name'].isin(temp_recommendations)]
        recommendations = recommend_based_on_filter(filtered_recommendations)
        

    response = {
        'method': 'Classification' if not any([price, ram, gpu]) else 'Keyword Extraction',
        'category': category if not any([price, ram, gpu]) else None,
        'confidence': confidence if not any([price, ram, gpu]) else None,
        'price': price if any([price, ram, gpu]) else None,
        'ram': ram if any([price, ram, gpu]) else None,
        'gpu': gpu if any([price, ram, gpu]) else None,
        'recommendations': recommendations
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()
