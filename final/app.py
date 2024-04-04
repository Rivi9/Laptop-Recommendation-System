from flask import Flask, render_template, request, jsonify
import pandas as pd
from advance_rec_dropdown import *
import pickle

app = Flask(__name__)
# Load laptop data from CSV
laptops_df = pd.read_csv("laptops_cleaned.csv")

@app.route('/', methods=['GET', 'POST'])
def start():
    return render_template('page1Select.html')

@app.route('/select_dropdown', methods=['POST'])
def select_dropdown():
    return render_template('recommendationPage1.html')

@app.route('/select_sentence', methods=['POST'])
def select_sentence():
    return render_template('sentence.html')

@app.route('/advance_rec_drop', methods=['GET', 'POST'])
def advance_rec_drop():
    if request.method == 'POST':
        price = request.form.get("priceRange")
        ram = request.form.get("ram")
        gpu = request.form.get("gpu")
        advance_rec_dropdown(price, ram, gpu)
        recommendation = advance_rec_dropdown(price, ram, gpu)
        return render_template('recommendations2.html', recommendation=recommendation)



@app.route('/backToRecPage1', methods=['GET', 'POST'])
def backToRecPage1():
    return render_template('recommendationPage1.html')

# Load the Word2Vec model
new = pickle.load(open('dataframe.pkl', 'rb'))
word2vec_similarity = pickle.load(open('word2vec_similarity.pkl', 'rb'))

# Function to recommend laptops
def recommend(use):
    index = new[new['usecases'] == use].index[0]
    distances = sorted(list(enumerate(word2vec_similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_laptops = []
    for i in distances[1:10]:
        row_index = i[0]
        name = new.iloc[row_index]['name']
        price = new.iloc[row_index]['price']
        img_link = new.iloc[row_index]['img_link']
        recommended_laptops.append({'name': name, 'price': price, 'img_link': img_link})
    return recommended_laptops

@app.route('/toRecPage2', methods=['GET', 'POST'])
def backToRecPage2():
    return render_template('recommendationPage2.html')

@app.route('/backToSelectPage', methods=['GET', 'POST'])
def backToSelectPage():
    return render_template('page1Select.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    use = request.form['use']
    recommendations = recommend(use)
    return render_template('recommendations.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run()