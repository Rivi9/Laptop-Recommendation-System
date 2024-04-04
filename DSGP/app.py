from flask import Flask, render_template, request, jsonify
import pandas as pd
from advance_rec_dropdown import *

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
        return render_template('recommendationPage2.html', price=price)



@app.route('/backToRecPage1', methods=['GET', 'POST'])
def backToRecPage1():
    return render_template('recommendationPage1.html')

#

@app.route('/toRecPage2', methods=['GET', 'POST'])
def backToRecPage2():
    return render_template('recommendationPage2.html')

@app.route('/backToSelectPage', methods=['GET', 'POST'])
def backToSelectPage():
    return render_template('page1Select.html')




if __name__ == '__main__':
    app.run()