import pandas as pd
import numpy as np
import spacy
import fasttext
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 

laptops_df = pd.read_csv("laptops_cleaned.csv")

# Load the Word2Vec model
new = pickle.load(open('dataframe.pkl', 'rb'))
word2vec_similarity = pickle.load(open('word2vec_similarity.pkl', 'rb'))

# Load spaCy model for keyword extraction
nlp = spacy.load("en_core_web_lg")

# Load FastText model for sentence classification
model = fasttext.load_model("input_classification.bin")

# Set of stopwords
stop_words = set(stopwords.words('english'))

# Mapping between predicted categories and dataset use cases
category_mapping = {
    "student_education": "Student/Education",
    "gaming": "Gaming",
    "basic": "Basic",
    "it": "IT",
    "business_professional": "Business/Professional",
    "creative_design": "Creative/Design"
}

# Define price range mappings
price_range_mappings = {
    "Less than $500": (None, 500),
    "$500 - $1000": (500, 1000),
    "$1000 - $1500": (1000, 1500),
    "More than $1500": (1500, None),
}


# Recommendation function
def recommend_sentence(use):
    mapped_use = category_mapping.get(use, use)  # Apply category mapping

    # Check if the use case exists in the dataframe
    if mapped_use not in new['usecases'].values:
        return "Use case not found. Please try a different one."

    index = new[new['usecases'] == mapped_use].index[0]
    distances = sorted(list(enumerate(word2vec_similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_laptops = []
    for i in distances[1:1000]:  # top 10 recommendations
        row_index = i[0]
        name = new.iloc[row_index]['name']
        price = new.iloc[row_index]['price']
        img_link = new.iloc[row_index]['img_link']
        recommended_laptops.append({'name': name, 'price': price, 'img_link': img_link})
    return recommended_laptops


def preprocess_text(text):
    # Preprocess the text using spaCy to extract keywords.
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
    stemmer = PorterStemmer()
    # Preprocess text for classification
    text = re.sub(r'[^\w\s]', '', text).lower()
    text = re.sub(' +', ' ', text)
    word_tokens = word_tokenize(text)
    filtered_text = [stemmer.stem(word) for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


def predict_category(sentence):
    # Predict the category using the FastText model
    preprocessed_sentence = preprocess_with_stemming(sentence)
    prediction = model.predict(preprocessed_sentence)
    # Remove the "__label__" prefix and convert the category to lowercase
    category = prediction[0][0].replace("__label__", "").lower()
    confidence = prediction[1][0]
    return category, confidence


def filter_laptops(price=None, ram=None, gpu=None, priceRange=None):
    filtered_df = laptops_df.copy()

    # Filter by price range if selected
    if priceRange:
        range_min, range_max = price_range_mappings.get(priceRange, (None, None))
        if range_min is not None:
            filtered_df = filtered_df[filtered_df['Price'] >= range_min]
        if range_max is not None:
            filtered_df = filtered_df[filtered_df['Price'] <= range_max]
    else:
        # Filter by specified price if no price range selected
        if price is not None:
            filtered_df = filtered_df[filtered_df['Price'] <= price]

    if ram is not None:
        ram_value = int(ram.replace("GB", ""))  # Convert input to integer
        filtered_df = filtered_df[filtered_df['Ram'].astype(int) >= ram_value]

    if gpu is not None:
        filtered_df = filtered_df[filtered_df['processor'].apply(lambda x: gpu.lower() in x.lower())]

    return filtered_df


def recommend_based_on_filter(filtered_df):
    recommendations = []
    for _, row in filtered_df.iterrows():
        company = row['Company']
        name = row['Product']
        gpu = row['Gpu']
        ram = row['Ram']
        price = row['Price']
        recommendations.append({'Company': company, 'Product': name, 'Gpu': gpu, 'Ram': ram, 'Price': price})
        if len(recommendations) == 10:  # top 10 recommendations
            break

    return recommendations


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    else:
        return obj
    
