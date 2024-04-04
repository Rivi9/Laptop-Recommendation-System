from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('recommendationPage1.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    use = request.form['use']
    recommendations = recommend(use)
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)


#Flask API Code