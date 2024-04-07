from flask import Flask, render_template, request, jsonify
import pandas as pd
from advance_rec_dropdown import *
import pickle
from sentence_rec import *

app = Flask(__name__)
# Load laptop data from CSV
laptops_df = pd.read_csv("laptops_cleaned.csv")

# Load the Word2Vec model
new = pickle.load(open('dataframe.pkl', 'rb'))
word2vec_similarity = pickle.load(open('word2vec_similarity.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('landingPage.html')

@app.route('/select_dropdown', methods=['POST'])
def select_dropdown():
    return render_template('recommendationPage1.html')

@app.route('/getStarted', methods=['POST'])
def get_started():
    return render_template('page1Select.html')

@app.route('/about_us', methods=['POST'])
def about_us():
    return render_template('aboutUs.html')

@app.route('/select_sentence', methods=['POST'])
def select_sentence():
    return render_template('sentence.html')

@app.route('/advance_rec_drop', methods=['GET', 'POST'])
def advance_rec_drop():
    if request.method == 'POST':
        price = request.form.get("priceRange")
        ram = request.form.get("ram")
        gpu = request.form.get("gpu")
        cpu = request.form.get("cpu")
        advance_rec_dropdown(price, ram, gpu, cpu)
        recommendation = advance_rec_dropdown(price, ram, gpu, cpu)
        return render_template('recommendations2.html', recommendation=recommendation)


@app.route('/backToRecPage1', methods=['GET', 'POST'])
def backToRecPage1():
    return render_template('recommendationPage1.html')

# Function to recommend laptops
def recommend(use):
    index = new[new['usecases'] == use].index[0]
    distances = sorted(list(enumerate(word2vec_similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_laptops = []
    for i in distances[1:1000]:
        row_index = i[0]
        name = new.iloc[row_index]['name']
        price = new.iloc[row_index]['price']
        img_link = new.iloc[row_index]['img_link']
        recommended_laptops.append({'name': name, 'price': price, 'img_link': img_link})
    return recommended_laptops


@app.route('/toRecPage2', methods=['GET', 'POST'])
def backToRecPage2():
    gpus_list = laptops_df['Gpu'].unique().tolist()
    gpus_list = sorted(gpus_list)
    cpus_list = laptops_df['Cpu'].unique().tolist()
    cpus_list = sorted(cpus_list)
    return render_template('recommendationPage2.html', gpus_list=gpus_list, cpus_list=cpus_list)

@app.route('/backToSelectPage', methods=['GET', 'POST'])
def backToSelectPage():
    return render_template('page1Select.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    use = request.form['use']
    priceRange = request.form.get('priceRange', None)  # Extract price range from the request

    recommendations = []  # Initialize recommendations list
    
    # Initially get recommendations without considering the price range
    initial_recommendations = recommend(use)

    # Filter recommendations based on the selected price range
    if priceRange:
        range_min, range_max = price_range_mappings.get(priceRange, (None, None))
        filtered_recommendations = []
        for rec in initial_recommendations:
            if range_min is not None and rec['price'] < range_min:
                continue  
            if range_max is not None and rec['price'] > range_max:
                continue  
            filtered_recommendations.append(rec)
            if len(filtered_recommendations) == 10:  # Break after adding the 10th recommendation
                break
        recommendations = filtered_recommendations
    else:
        recommendations = initial_recommendations[:10]
        
    return render_template('recommendations.html', recommendations=recommendations)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentence = data['sentence']
    priceRange = data.get('priceRange', None)  # Extract price range from the request

    # Extract keywords first
    price, ram, gpu = preprocess_text(sentence)
    recommendations = []  # Initialize recommendations list

    # extracted keywords for recommendation
    if any([price, ram, gpu]):
        # If specific keywords were found - use keywords for recommendations
        filtered_df = filter_laptops(price=price, ram=ram, gpu=gpu, priceRange=priceRange)
        recommendations = recommend_based_on_filter(filtered_df)

    else:
        # No specific keywords - use the classification model
        category, confidence = predict_category(sentence)

        # Initially get recommendations without considering the price range
        initial_recommendations = recommend_sentence(category)

        # Filter recommendations based on the selected price range
        if priceRange:
            range_min, range_max = price_range_mappings.get(priceRange, (None, None))
            filtered_recommendations = []
            for rec in initial_recommendations:
                if range_min is not None and rec['price'] < range_min:
                    continue
                if range_max is not None and rec['price'] > range_max:
                    continue
                filtered_recommendations.append(rec)
                if len(filtered_recommendations) == 10:  # Break after adding the 10th recommendation
                    break
            recommendations = filtered_recommendations
        else:
            recommendations = initial_recommendations[:10]

    response = {
        'method': 'Classification' if not any([price, ram, gpu]) else 'Keyword Extraction',
        'category': category if not any([price, ram, gpu]) else None,
        'confidence': confidence if not any([price, ram, gpu]) else None,
        'price': price if any([price, ram, gpu]) else None,
        'ram': ram if any([price, ram, gpu]) else None,
        'gpu': gpu if any([price, ram, gpu]) else None,
        'recommendations': recommendations
    }

    response = convert_numpy(response)  # Convert numpy types to native Python types
    return jsonify(response)


if __name__ == '__main__':
    app.run()
